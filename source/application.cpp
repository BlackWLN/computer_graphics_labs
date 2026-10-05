// application.cpp - тут живет вся лабораторная.
//
// стартовый код (main.cpp и graphics_internal.cpp) поднимает окно, vulkan и imgui,
// а дальше сам дергает наши функции (они в самом низу файла):
//   initialize() - один раз при старте: создаем буферы, дескрипторы и пайплайны
//   update()     - каждый кадр перед отрисовкой: двигаем анимацию и рисуем окно настроек
//   render()     - каждый кадр: заливаем данные в буферы и записываем команды отрисовки
//   shutdown()   - при выходе: все удаляем
// все, что выше этих функций, - помощники для них.

#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include "geometry.hpp"
#include "math.hpp"

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

#ifndef ASSETS_DIR
#define ASSETS_DIR "assets"
#endif

namespace application {

namespace {

using math::Mat4;
using math::Vec3;

//  данные, которые улетают в шейдеры

// uniform-буфер кадра (set = 0) - общий для всех объектов: камера, проекция, свет.
// поля идут кусками по 16 байт (mat4 и vec4), поэтому раскладка в c++ совпадает с std140 в glsl
struct FrameUniforms {
	Mat4 view;
	Mat4 projection;
	float light[4];   // направление на свет + доля фонового света
	float options[4]; // x - освещение вкл/выкл
};

// uniform-буфер объекта (set = 1) - у каждого объекта свой буфер и свой
// VkDescriptorSet. это и есть доп. задание 6
struct ObjectUniforms {
	Mat4 model;
	float color[4];   // цвет объекта
	float params[4];  // x - 1: умножать на цвета вершин, 0: не умножать
};

// push-константа - маленький кусочек данных, который едет прямо в команде отрисовки
struct DrawConstants {
	float params[4]; // x - 0: рисуем грани, 1: рисуем линии
};

//  состояние сцены - все, что крутится ползунками в интерфейсе

enum class Projection : int { Perspective = 0, Orthographic = 1 };
enum class Scene : int { Single = 0, PendulumWave = 1 };

struct State {
	// доп. задание 1: проекция
	Projection projection = Projection::Perspective;
	float fov = 60.0f;             // градусы
	float near_plane = 0.1f;
	float far_plane = 100.0f;

	// орбитальная камера - летает по сфере вокруг начала координат
	float camera_yaw = 35.0f;
	float camera_pitch = 25.0f;
	float camera_distance = 8.0f;

	// форма параллелепипеда
	geometry::ParallelepipedParams shape;
	bool mesh_dirty = true;  // true -> надо пересобрать вершины

	// доп. задание 2: положение, поворот, масштаб
	Vec3 position{0.0f, 0.0f, 0.0f};
	Vec3 rotation{0.0f, 0.0f, 0.0f};   // углы эйлера в градусах
	Vec3 scale{1.0f, 1.0f, 1.0f};
	bool uniform_scale = false;

	// доп. задание 3: анимация
	bool move_along_path = true;
	bool playing = true;
	float speed = 0.6f;
	float anim_time = 0.0f;            // свое время анимации, тикает только когда не на паузе
	geometry::TrajectoryParams trajectory;
	bool show_path = true;
	bool follow_tangent = true;
	float spin_speed = 60.0f;          // градусов в секунду
	int spin_axis = 1;                 // 0 - x, 1 - y, 2 - z
	float spin_angle = 0.0f;

	// доп. задания 4 и 5: цвет
	float color[3] = {1.0f, 1.0f, 1.0f};
	geometry::ColorMode color_mode = geometry::ColorMode::PositionRGB;
	bool lighting = true;
	bool show_edges = true;
	float background[3] = {0.11f, 0.12f, 0.15f};

	// доп. задание 6: вторая сцена - маятниковая волна
	Scene scene = Scene::Single;
	int pendulum_count = 15;
	float pendulum_cycle = 30.0f;      // через столько секунд маятники снова встают в линию
	int pendulum_swings = 20;          // сколько раз качнется самый длинный маятник за цикл
	float pendulum_amplitude = 25.0f;  // градусы
	float pendulum_length = 3.5f;      // длина самого длинного маятника
	bool pendulum_playing = true;
	float pendulum_speed = 1.0f;
	float pendulum_time = 0.0f;

	double last_time = -1.0;
};

State state;

//  vulkan-ресурсы

// буфер в памяти, которую видит процессор: пишем в него обычным memcpy по готовому указателю
struct HostBuffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
	VkDeviceSize capacity = 0;
};

// объект сцены = свой uniform-буфер + свой набор дескрипторов, который на него смотрит
struct SceneObject {
	HostBuffer uniforms;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

constexpr int max_pendulums = 24;
constexpr int object_main = 0;            // параллелепипед из основной сцены
constexpr int object_world = 1;           // линии в мировых координатах: траектория, нити, перекладина
constexpr int object_first_pendulum = 2;  // дальше идут маятники
constexpr int object_count = object_first_pendulum + max_pendulums;

VkDescriptorSetLayout descriptor_set_layout; // один layout на все наборы: и для кадра, и для объектов
VkDescriptorPool descriptor_pool;
VkDescriptorSet frame_descriptor_set;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline_triangles;
VkPipeline pipeline_lines;

HostBuffer frame_uniforms;
SceneObject objects[object_count];
HostBuffer mesh_buffer;
HostBuffer edges_buffer;
HostBuffer lines_buffer; // траектория или нити маятников

uint32_t mesh_vertex_count = 0;
uint32_t edges_vertex_count = 0;
uint32_t lines_vertex_count = 0;

constexpr int path_segments = 400;
constexpr int lines_capacity = 2 * path_segments + 2 * (max_pendulums + 1);

// создает буфер через vma в памяти, которую видит процессор, и сразу дает на нее указатель.
// данных у нас мало и они меняются каждый кадр, поэтому так проще, чем через staging-буфер
bool createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage, HostBuffer& out) {
	auto& context = graphics::internal::context;

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo info{};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &out.buffer, &out.allocation, &info) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer\n";
		return false;
	}

	out.mapped = info.pMappedData;
	out.capacity = size;
	return true;
}

void destroyHostBuffer(HostBuffer& b) {
	if (b.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, b.buffer, b.allocation);
	}
	b = {};
}

// копирует данные в буфер
void uploadToBuffer(HostBuffer& b, const void* data, VkDeviceSize size) {
	size = std::min(size, b.capacity);
	std::memcpy(b.mapped, data, size_t(size));
	// если память не host_coherent, надо сказать gpu, что мы ее поменяли
	vmaFlushAllocation(graphics::internal::context.allocator, b.allocation, 0, size);
}

std::vector<char> readFile(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		return {};
	}
	std::vector<char> data(size_t(file.tellg()));
	file.seekg(0);
	file.read(data.data(), std::streamsize(data.size()));
	return data;
}

// читает скомпилированный шейдер (.spv) и делает из него VkShaderModule
VkShaderModule loadShader(const char* path) {
	const std::vector<char> code = readFile(path);
	if (code.empty()) {
		std::cerr << "Failed to read shader " << path << '\n';
		return VK_NULL_HANDLE;
	}

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data()),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module " << path << '\n';
	}
	return module;
}

// берет набор дескрипторов из пула и направляет его на наш uniform-буфер
VkDescriptorSet allocateDescriptorSet(const HostBuffer& buffer, VkDeviceSize size) {
	auto& context = graphics::internal::context;

	const VkDescriptorSetAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	VkDescriptorSet set = VK_NULL_HANDLE;
	if (vkAllocateDescriptorSets(context.device, &alloc_info, &set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return VK_NULL_HANDLE;
	}

	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer.buffer,
		.offset = 0,
		.range = size,
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	return set;
}

// создаем layout, пул и сами наборы дескрипторов: один на кадр и по одному на каждый объект
bool createDescriptors() {
	auto& context = graphics::internal::context;

	// layout: одна привязка - uniform-буфер, который видят оба шейдера.
	// подходит и для набора кадра (set = 0), и для наборов объектов (set = 1)
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr,
	                                &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	// пул: 1 набор на кадр + по набору на каждый объект
	const uint32_t set_count = 1 + object_count;

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = set_count,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = set_count,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	frame_descriptor_set = allocateDescriptorSet(frame_uniforms, sizeof(FrameUniforms));
	if (frame_descriptor_set == VK_NULL_HANDLE) {
		return false;
	}

	for (SceneObject& object : objects) {
		object.descriptor_set = allocateDescriptorSet(object.uniforms, sizeof(ObjectUniforms));
		if (object.descriptor_set == VK_NULL_HANDLE) {
			return false;
		}
	}

	return true;
}

// собирает графический пайплайн. шейдеры у граней и линий одни и те же,
// отличаются топология (треугольники или отрезки) и настройки растеризации
VkPipeline createPipeline(VkShaderModule vert, VkShaderModule frag,
                          VkPrimitiveTopology topology, bool is_lines) {
	auto& context = graphics::internal::context;

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = frag,
			.pName = "main",
		},
	};

	// формат вершины: один буфер, три атрибута (позиция, нормаль, цвет)
	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(geometry::Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attributes[] = {
		{.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		 .offset = offsetof(geometry::Vertex, position)},
		{.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		 .offset = offsetof(geometry::Vertex, normal)},
		{.location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		 .offset = offsetof(geometry::Vertex, color)},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = 3,
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology,
	};

	// viewport и scissor задаем динамически, чтобы не пересоздавать
	// пайплайн каждый раз, когда меняется размер окна
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	// для граней включаем отсечение задних граней и чуть-чуть сдвигаем глубину назад (depth bias), чтобы ребра поверх граней не мерцали (z-fighting)
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VkCullModeFlags(is_lines ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT),
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = is_lines ? VK_FALSE : VK_TRUE,
		.depthBiasConstantFactor = 1.0f,
		.depthBiasSlopeFactor = 1.0f,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
	};

	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	VkPipeline pipeline = VK_NULL_HANDLE;
	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr,
	                              &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
	}
	return pipeline;
}

// pipeline layout (какие наборы и push-константы ждут шейдеры) + два пайплайна
bool createPipelines() {
	auto& context = graphics::internal::context;

	const VkPushConstantRange push_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(DrawConstants),
	};

	// set = 0 - данные кадра, set = 1 - данные объекта (layout у них одинаковый)
	const VkDescriptorSetLayout set_layouts[] = {descriptor_set_layout, descriptor_set_layout};

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	VkShaderModule vert = loadShader(SHADER_DIR "/mesh.vert.spv");
	VkShaderModule frag = loadShader(SHADER_DIR "/mesh.frag.spv");
	if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
		return false;
	}

	pipeline_triangles = createPipeline(vert, frag, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false);
	pipeline_lines = createPipeline(vert, frag, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, true);

	// шейдерные модули нужны только пока создаем пайплайн, потом их можно удалить
	vkDestroyShaderModule(context.device, vert, nullptr);
	vkDestroyShaderModule(context.device, frag, nullptr);

	return pipeline_triangles != VK_NULL_HANDLE && pipeline_lines != VK_NULL_HANDLE;
}

// пересобирает вершины параллелепипеда (поменяли форму или раскраску) и заливает их в буферы
void rebuildMesh() {
	const geometry::Mesh mesh = geometry::buildParallelepiped(state.shape, state.color_mode);

	uploadToBuffer(mesh_buffer, mesh.triangles.data(), mesh.triangles.size() * sizeof(geometry::Vertex));
	uploadToBuffer(edges_buffer, mesh.edges.data(), mesh.edges.size() * sizeof(geometry::Vertex));

	mesh_vertex_count = uint32_t(mesh.triangles.size());
	edges_vertex_count = uint32_t(mesh.edges.size());
	state.mesh_dirty = false;
}

//  матрицы

// позиция камеры на сфере вокруг центра: азимут, высота, расстояние -> x, y, z
Vec3 cameraPosition() {
	const float yaw = math::radians(state.camera_yaw);
	const float pitch = math::radians(state.camera_pitch);
	return Vec3{
		std::cos(pitch) * std::sin(yaw),
		std::sin(pitch),
		std::cos(pitch) * std::cos(yaw),
	} * state.camera_distance;
}

// доп. задание 1: перспективная или ортографическая матрица
Mat4 projectionMatrix(float aspect) {
	if (state.projection == Projection::Perspective) {
		return math::perspective(math::radians(state.fov), aspect, state.near_plane, state.far_plane);
	}

	// размер ортографического окна подбираем так, чтобы в центре сцены масштаб
	// совпадал с перспективой - тогда при переключении картинка не прыгает.
	// высота окна = 2 * d * tan(fov / 2)
	const float half_h = state.camera_distance * std::tan(math::radians(state.fov) * 0.5f);
	const float half_w = half_h * aspect;
	return math::orthographic(-half_w, half_w, -half_h, half_h, state.near_plane, state.far_plane);
}

// доп. задания 2 и 3: собираем матрицу модели параллелепипеда
Mat4 modelMatrix() {
	Vec3 translation = state.position;
	Mat4 orientation = Mat4::identity();

	if (state.move_along_path) {
		translation = translation + geometry::trajectoryPoint(state.trajectory, state.anim_time);

		if (state.follow_tangent) {
			// разворачиваем фигуру, чтобы ее локальная ось x смотрела туда, куда она летит (по касательной к траектории)
			const Vec3 forward = geometry::trajectoryTangent(state.trajectory, state.anim_time);
			Vec3 up{0.0f, 1.0f, 0.0f};
			if (std::fabs(math::dot(forward, up)) > 0.99f) {
				up = {1.0f, 0.0f, 0.0f};
			}
			const Vec3 side = math::normalize(math::cross(forward, up));
			const Vec3 true_up = math::cross(side, forward);
			orientation = math::basis(forward, true_up, math::cross(forward, true_up));
		}
	}

	const float spin = math::radians(state.spin_angle);
	const Mat4 spin_matrix = state.spin_axis == 0 ? math::rotateX(spin)
	                       : state.spin_axis == 1 ? math::rotateY(spin)
	                                              : math::rotateZ(spin);

	const Vec3 euler{
		math::radians(state.rotation.x),
		math::radians(state.rotation.y),
		math::radians(state.rotation.z),
	};

	// M = T * R_траектории * R_вращения * R_пользователя * S
	// читаем справа налево: сначала масштаб, потом повороты, в конце перенос
	return math::translate(translation) * orientation * spin_matrix *
	       math::rotateEuler(euler) * math::scale(state.scale);
}

// маятниковая волна (доп. задание 6) - прикол с несколькими объектами 
// идея: за цикл T самый длинный маятник качается n раз, следующий n+1, n+2 и т.д.
// через T секунд все снова в одной фазе и встают в линию, а между этим рисуют змейку.
// частота i-го: w_i = 2*pi*(n + i) / T. длину берем из формулы периода маятника
// T = 2*pi*sqrt(L/g): L ~ 1/w^2, значит L_i = L_max * (n / (n + i))^2
constexpr float pendulum_top = 2.0f;      // высота перекладины
constexpr float pendulum_spacing = 0.55f; // расстояние между маятниками

// точка подвеса i-го маятника - маятники стоят в ряд вдоль оси x
Vec3 pendulumPivot(int i) {
	const float x = (float(i) - float(state.pendulum_count - 1) * 0.5f) * pendulum_spacing;
	return {x, pendulum_top, 0.0f};
}

// матрица i-го грузика, а заодно его центр. он нужен, чтобы нарисовать нить)
Mat4 pendulumMatrix(int i, Vec3& bob_center) {
	const float n = float(state.pendulum_swings);
	const float omega = 2.0f * math::pi * (n + float(i)) / state.pendulum_cycle;
	const float length = state.pendulum_length * (n / (n + float(i))) * (n / (n + float(i)));
	const float angle = math::radians(state.pendulum_amplitude) * std::cos(omega * state.pendulum_time);

	const Vec3 pivot = pendulumPivot(i);
	// поворот вокруг оси x переводит точку (0, -L, 0) в (0, -L*cos(a), -L*sin(a))
	bob_center = pivot + Vec3{0.0f, -length * std::cos(angle), -length * std::sin(angle)};

	// цепочка преобразований: перенос в точку подвеса -> поворот на угол -> сдвиг вниз на длину нити -> уменьшаем параллелепипед до размера грузика
	return math::translate(pivot) * math::rotateX(angle) *
	       math::translate({0.0f, -length, 0.0f}) * math::scale({0.2f, 0.2f, 0.2f});
}

// записывает матрицу и цвет в uniform-буфер объекта
void writeObject(int index, const Mat4& model, const float color[3], bool use_vertex_colors) {
	ObjectUniforms data{};
	data.model = model;
	data.color[0] = color[0];
	data.color[1] = color[1];
	data.color[2] = color[2];
	data.color[3] = 1.0f;
	data.params[0] = use_vertex_colors ? 1.0f : 0.0f;
	uploadToBuffer(objects[index].uniforms, &data, sizeof(data));
}

//  интерфейс (imgui)

// мышка и тачпад!! тянем левой кнопкой (одним тапом) - крутим камеру, колесо (или двумя пальцами водим) - приближаем
void handleMouse() {
	ImGuiIO& io = ImGui::GetIO();
	if (io.WantCaptureMouse) {
		return; // мышь над окном imgui - камеру не трогаем
	}

	if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
		state.camera_yaw -= io.MouseDelta.x * 0.4f;
		if (state.camera_yaw > 180.0f) state.camera_yaw -= 360.0f;
		if (state.camera_yaw < -180.0f) state.camera_yaw += 360.0f;
		state.camera_pitch = std::clamp(state.camera_pitch + io.MouseDelta.y * 0.4f, -89.0f, 89.0f);
	}

	if (io.MouseWheel != 0.0f) {
		state.camera_distance = std::clamp(state.camera_distance * std::pow(0.9f, io.MouseWheel), 2.0f, 50.0f);
	}
}

// окно с настройками. imgui работает так: каждый кадр мы заново описываем все элементы,
// а он сам их рисует и меняет наши переменные, когда двигаешь ползунок
void drawInterface() {
	ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(400, ImGui::GetIO().DisplaySize.y - 20.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Параметры");

	ImGui::TextDisabled("ЛКМ по сцене — вращать камеру, колесо — масштаб");

	int scene = int(state.scene);
	ImGui::RadioButton("Один параллелепипед", &scene, int(Scene::Single));
	ImGui::SameLine();
	ImGui::RadioButton("Маятниковая волна##scene", &scene, int(Scene::PendulumWave));
	if (scene != int(state.scene)) {
		state.scene = Scene(scene);
		if (state.scene == Scene::PendulumWave) {
			state.pendulum_time = 0.0f; // начинаем с ровной линии
			state.camera_yaw = 25.0f;
			state.camera_pitch = 20.0f;
			state.camera_distance = 10.0f;
		}
	}
	const bool pendulums = state.scene == Scene::PendulumWave;

	//доп. задание 1: проекция
	if (ImGui::CollapsingHeader("Проекция", ImGuiTreeNodeFlags_DefaultOpen)) {
		int p = int(state.projection);
		ImGui::RadioButton("Перспективная", &p, int(Projection::Perspective));
		ImGui::SameLine();
		ImGui::RadioButton("Ортографическая", &p, int(Projection::Orthographic));
		state.projection = Projection(p);

		ImGui::SliderFloat("Угол обзора", &state.fov, 20.0f, 120.0f, "%.0f°");
		ImGui::SliderFloat("Расстояние", &state.camera_distance, 2.0f, 50.0f, "%.1f");
		ImGui::SliderFloat("Азимут", &state.camera_yaw, -180.0f, 180.0f, "%.0f°");
		ImGui::SliderFloat("Высота", &state.camera_pitch, -89.0f, 89.0f, "%.0f°");
	}

	// форма параллелепипеда
	if (ImGui::CollapsingHeader("Параллелепипед")) {
		geometry::ParallelepipedParams edited = state.shape;
		bool changed = false;
		changed |= ImGui::SliderFloat("Ребро a", &edited.a, 0.2f, 4.0f, "%.2f");
		changed |= ImGui::SliderFloat("Ребро b", &edited.b, 0.2f, 4.0f, "%.2f");
		changed |= ImGui::SliderFloat("Ребро c", &edited.c, 0.2f, 4.0f, "%.2f");
		changed |= ImGui::SliderFloat("α (b, c)", &edited.alpha, 20.0f, 160.0f, "%.0f°");
		changed |= ImGui::SliderFloat("β (a, c)", &edited.beta, 20.0f, 160.0f, "%.0f°");
		changed |= ImGui::SliderFloat("γ (a, b)", &edited.gamma, 20.0f, 160.0f, "%.0f°");

		if (ImGui::Button("Прямоугольный")) {
			edited.alpha = edited.beta = edited.gamma = 90.0f;
			changed = true;
		}
		ImGui::SameLine();
		if (ImGui::Button("По умолчанию")) {
			edited = {};
			changed = true;
		}

		// вообще не любые три угла дают параллелепипед - объем должен быть больше нуля
		if (changed) {
			if (geometry::volumeFactor(edited) > 1e-3f) {
				state.shape = edited;
				state.mesh_dirty = true;
			}
		}

		const float volume = state.shape.a * state.shape.b * state.shape.c *
		                     std::sqrt(std::max(0.0f, geometry::volumeFactor(state.shape)));
		ImGui::Text("Объём V = %.3f", volume);
	}

	// доп. задание 6: маятники
	if (pendulums && ImGui::CollapsingHeader("Маятниковая волна", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button(state.pendulum_playing ? "Пауза##p" : "Воспроизвести##p", ImVec2(130, 0))) {
			state.pendulum_playing = !state.pendulum_playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("Запустить заново")) {
			state.pendulum_time = 0.0f;
		}

		ImGui::SliderFloat("Скорость##p", &state.pendulum_speed, 0.0f, 3.0f, "%.2f");
		ImGui::SliderInt("Количество", &state.pendulum_count, 3, max_pendulums);
		ImGui::SliderFloat("Амплитуда", &state.pendulum_amplitude, 5.0f, 45.0f, "%.0f°");
		ImGui::SliderFloat("Длина цикла", &state.pendulum_cycle, 10.0f, 90.0f, "%.0f с");
		ImGui::SliderInt("Колебаний за цикл", &state.pendulum_swings, 5, 40);

		char progress[32];
		std::snprintf(progress, sizeof(progress), "%.1f / %.0f с", state.pendulum_time, state.pendulum_cycle);
		ImGui::ProgressBar(state.pendulum_time / state.pendulum_cycle, ImVec2(-1, 0), progress);

		if (ImGui::Button("Вид сверху")) {
			state.camera_yaw = 0.0f;
			state.camera_pitch = 89.0f;
		}
		ImGui::SameLine();
		if (ImGui::Button("Вид сбоку")) {
			state.camera_yaw = 90.0f;
			state.camera_pitch = 5.0f;
		}
		ImGui::SameLine();
		if (ImGui::Button("Вид 3/4")) {
			state.camera_yaw = 25.0f;
			state.camera_pitch = 20.0f;
		}
	}

	//доп. задание 2: положение, поворот, масштаб
	if (!pendulums && ImGui::CollapsingHeader("Положение, поворот, масштаб", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Позиция", &state.position.x, 0.02f, -10.0f, 10.0f, "%.2f");
		ImGui::SliderFloat3("Поворот XYZ", &state.rotation.x, -180.0f, 180.0f, "%.0f°");

		ImGui::Checkbox("Равномерный масштаб", &state.uniform_scale);
		if (state.uniform_scale) {
			if (ImGui::SliderFloat("Масштаб", &state.scale.x, 0.1f, 3.0f, "%.2f")) {
				state.scale.y = state.scale.z = state.scale.x;
			}
		} else {
			ImGui::SliderFloat3("Масштаб XYZ", &state.scale.x, 0.1f, 3.0f, "%.2f");
		}

		if (ImGui::Button("Сбросить преобразования")) {
			state.position = {};
			state.rotation = {};
			state.scale = {1.0f, 1.0f, 1.0f};
		}
	}

	// ---------- доп. задание 3: анимация ----------
	if (!pendulums && ImGui::CollapsingHeader("Анимация", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button(state.playing ? "Пауза" : "Воспроизвести", ImVec2(130, 0))) {
			state.playing = !state.playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("В начало")) {
			state.anim_time = 0.0f;
			state.spin_angle = 0.0f;
		}

		ImGui::SliderFloat("Скорость", &state.speed, 0.0f, 3.0f, "%.2f");

		ImGui::SeparatorText("Траектория");
		ImGui::Checkbox("Двигаться по траектории", &state.move_along_path);

		const char* names[] = {"Окружность", "Фигура Лиссажу", "Узел-трилистник", "Роза"};
		int type = int(state.trajectory.type);
		ImGui::Combo("Тип", &type, names, IM_ARRAYSIZE(names));
		state.trajectory.type = geometry::Trajectory(type);

		ImGui::SliderFloat("Радиус", &state.trajectory.radius, 0.5f, 6.0f, "%.2f");
		if (state.trajectory.type != geometry::Trajectory::Circle) {
			ImGui::SliderFloat("Амплитуда по высоте", &state.trajectory.height, 0.0f, 3.0f, "%.2f");
		}
		if (state.trajectory.type == geometry::Trajectory::Lissajous) {
			ImGui::SliderInt("Частота X", &state.trajectory.freq_a, 1, 5);
			ImGui::SliderInt("Частота Z", &state.trajectory.freq_b, 1, 5);
			ImGui::SliderFloat("Сдвиг фаз", &state.trajectory.phase, 0.0f, 180.0f, "%.0f°");
		}
		if (state.trajectory.type == geometry::Trajectory::Rose) {
			ImGui::SliderInt("Лепестки k", &state.trajectory.freq_a, 1, 7);
		}

		ImGui::Checkbox("Показывать траекторию", &state.show_path);
		ImGui::Checkbox("Разворачивать по направлению движения", &state.follow_tangent);

		ImGui::SeparatorText("Вращение вокруг своей оси");
		ImGui::SliderFloat("Угловая скорость", &state.spin_speed, -360.0f, 360.0f, "%.0f°/с");
		ImGui::Combo("Ось", &state.spin_axis, "X\0Y\0Z\0");
	}

	//доп. задания 4 и 5: цвет
	if (ImGui::CollapsingHeader("Цвет", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Цвет фигуры", state.color);

		const char* modes[] = {"Координаты XYZ → RGB", "Градиент по высоте", "Оттенок вершины", "Без окраски (белый)"};
		int mode = int(state.color_mode);
		if (ImGui::Combo("Цвета вершин", &mode, modes, IM_ARRAYSIZE(modes))) {
			state.color_mode = geometry::ColorMode(mode);
			state.mesh_dirty = true;
		}
		ImGui::TextDisabled("Итоговый цвет = цвет фигуры × цвет вершины");

		ImGui::Checkbox("Освещение", &state.lighting);
		ImGui::SameLine();
		ImGui::Checkbox("Рёбра", &state.show_edges);
		ImGui::ColorEdit3("Фон", state.background);
	}

	ImGui::Separator();
	ImGui::Text("%.0f кадров/с", ImGui::GetIO().Framerate);

	ImGui::End();
}

}

//  функции, которые дергает main.cpp один раз при старте

bool initialize() {
	// шрифт с кириллицей - стандартный шрифт imgui умеет только латиницу
	ImGuiIO& io = ImGui::GetIO();
	if (io.Fonts->AddFontFromFileTTF(ASSETS_DIR "/fonts/DejaVuSans.ttf", 16.0f) == nullptr) {
		std::cerr << "Failed to load font, Cyrillic text will not be displayed\n";
	}
	ImGui::GetStyle().FrameRounding = 3.0f;
	ImGui::GetStyle().WindowRounding = 5.0f;

	if (!createHostBuffer(sizeof(FrameUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, frame_uniforms) ||
	    !createHostBuffer(64 * sizeof(geometry::Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh_buffer) ||
	    !createHostBuffer(64 * sizeof(geometry::Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, edges_buffer) ||
	    !createHostBuffer(lines_capacity * sizeof(geometry::Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, lines_buffer)) {
		return false;
	}

	for (SceneObject& object : objects) {
		if (!createHostBuffer(sizeof(ObjectUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, object.uniforms)) {
			return false;
		}
	}

	if (!createDescriptors() || !createPipelines()) {
		return false;
	}

	rebuildMesh();
	return true;
}

// при выходе удаляем все, что создали
void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline_lines, nullptr);
	vkDestroyPipeline(context.device, pipeline_triangles, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr); // наборы удаляются вместе с пулом
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);

	for (SceneObject& object : objects) {
		destroyHostBuffer(object.uniforms);
	}
	destroyHostBuffer(lines_buffer);
	destroyHostBuffer(edges_buffer);
	destroyHostBuffer(mesh_buffer);
	destroyHostBuffer(frame_uniforms);
}

// каждый кадр перед отрисовкой: двигаем время и рисуем интерфейс
void update(double time) {
	const float dt = state.last_time < 0.0 ? 0.0f : float(time - state.last_time);
	state.last_time = time;

	// время анимации копится только когда она играет, поэтому пауза и смена скорости не дают рывков
	if (state.playing) {
		state.anim_time = std::fmod(state.anim_time + dt * state.speed, 2.0f * math::pi);
		state.spin_angle = std::fmod(state.spin_angle + dt * state.speed * state.spin_speed, 360.0f);
	}

	if (state.pendulum_playing) {
		state.pendulum_time = std::fmod(state.pendulum_time + dt * state.pendulum_speed, state.pendulum_cycle);
	}

	handleMouse();
	drawInterface();
}

// каждый кадр: заливаем данные в буферы и записываем команды отрисовки
void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	// prepare() уже дождался fence прошлого кадра, значит gpu
	// больше не читает наши буферы и их можно спокойно перезаписывать
	if (state.mesh_dirty) {
		rebuildMesh();
	}

	const bool pendulums = state.scene == Scene::PendulumWave;
	const float white[3] = {1.0f, 1.0f, 1.0f};

	//1. данные объектов
	std::vector<geometry::Vertex> lines; // линии в мировых координатах

	if (!pendulums) {
		writeObject(object_main, modelMatrix(), state.color, true);

		if (state.show_path && state.move_along_path) {
			lines = geometry::trajectoryLines(state.trajectory, Vec3{0.95f, 0.85f, 0.35f}, path_segments);
			for (geometry::Vertex& v : lines) {
				v.position = v.position + state.position; // траектория крутится вокруг позиции фигуры
			}
		}
	} else {
		const Vec3 string_color{0.7f, 0.7f, 0.75f};

		// перекладина
		lines.push_back({pendulumPivot(0) - Vec3{0.3f, 0.0f, 0.0f}, {}, string_color});
		lines.push_back({pendulumPivot(state.pendulum_count - 1) + Vec3{0.3f, 0.0f, 0.0f}, {}, string_color});

		for (int i = 0; i < state.pendulum_count; ++i) {
			Vec3 bob;
			const Mat4 model = pendulumMatrix(i, bob);

			// радуга, умноженная на цвет из интерфейса
			const Vec3 rainbow = geometry::hsv(float(i) / float(state.pendulum_count), 0.7f, 1.0f);
			const float color[3] = {rainbow.x * state.color[0], rainbow.y * state.color[1], rainbow.z * state.color[2]};
			writeObject(object_first_pendulum + i, model, color, false);

			// нить от точки подвеса до грузика
			lines.push_back({pendulumPivot(i), {}, string_color});
			lines.push_back({bob, {}, string_color});
		}
	}

	writeObject(object_world, Mat4::identity(), white, true);
	uploadToBuffer(lines_buffer, lines.data(), lines.size() * sizeof(geometry::Vertex));
	lines_vertex_count = uint32_t(lines.size());

	//2. данные кадра (камера, проекция, свет)
	const VkExtent2D extent = context.swapchain_extent;
	const float aspect = float(extent.width) / float(std::max(extent.height, 1u));

	FrameUniforms frame{};
	frame.view = math::lookAt(cameraPosition(), Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f});
	frame.projection = projectionMatrix(aspect);
	const Vec3 light = math::normalize(Vec3{0.5f, 0.9f, 0.7f});
	frame.light[0] = light.x;
	frame.light[1] = light.y;
	frame.light[2] = light.z;
	frame.light[3] = 0.45f;
	frame.options[0] = state.lighting ? 1.0f : 0.0f;
	uploadToBuffer(frame_uniforms, &frame, sizeof(frame));

	//3. записываем команды в командный буфер
	const VkCommandBuffer cmd = fd.command_buffer;

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin);

	VkClearValue clear[2] = {};
	clear[0].color = {{state.background[0], state.background[1], state.background[2], 1.0f}};
	clear[1].depthStencil = {1.0f, 0};

	const VkRenderPassBeginInfo pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = {.offset = {0, 0}, .extent = extent},
		.clearValueCount = 2,
		.pClearValues = clear,
	};
	// начинаем render pass: экран заливается цветом фона, глубина сбрасывается в 1
	vkCmdBeginRenderPass(cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0.0f, .y = 0.0f,
		.width = float(extent.width), .height = float(extent.height),
		.minDepth = 0.0f, .maxDepth = 1.0f,
	};
	const VkRect2D scissor = {.offset = {0, 0}, .extent = extent};
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	const VkDeviceSize offset = 0;
	const VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	// какие объекты рисуем параллелепипедом
	std::vector<int> drawn;
	if (pendulums) {
		for (int i = 0; i < state.pendulum_count; ++i) {
			drawn.push_back(object_first_pendulum + i);
		}
	} else {
		drawn.push_back(object_main);
	}

	// набор 0 (данные кадра) привязываем один раз - он общий для всех
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_triangles);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &frame_descriptor_set, 0, nullptr);

	// грани: сетка одна и та же, но перед каждым объектом привязываем его набор 1
	DrawConstants dc{};
	dc.params[0] = 0.0f;
	vkCmdPushConstants(cmd, pipeline_layout, stages, 0, sizeof(dc), &dc);
	vkCmdBindVertexBuffers(cmd, 0, 1, &mesh_buffer.buffer, &offset);
	for (int index : drawn) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        1, 1, &objects[index].descriptor_set, 0, nullptr);
		vkCmdDraw(cmd, mesh_vertex_count, 1, 0, 0);
	}

	// линии (layout тот же, поэтому привязанные наборы никуда не деваются)
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_lines);
	dc.params[0] = 1.0f;
	vkCmdPushConstants(cmd, pipeline_layout, stages, 0, sizeof(dc), &dc);

	if (state.show_edges) {
		vkCmdBindVertexBuffers(cmd, 0, 1, &edges_buffer.buffer, &offset);
		for (int index : drawn) {
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
			                        1, 1, &objects[index].descriptor_set, 0, nullptr);
			vkCmdDraw(cmd, edges_vertex_count, 1, 0, 0);
		}
	}

	if (lines_vertex_count > 0) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        1, 1, &objects[object_world].descriptor_set, 0, nullptr);
		vkCmdBindVertexBuffers(cmd, 0, 1, &lines_buffer.buffer, &offset);
		vkCmdDraw(cmd, lines_vertex_count, 1, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

}