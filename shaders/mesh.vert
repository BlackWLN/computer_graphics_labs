#version 450

// вершинный шейдер: запускается для каждой вершины и считает, где она окажется на экране

// данные кадра - общие для всех объектов (набор дескрипторов 0)
layout(set = 0, binding = 0) uniform FrameUniforms {
	mat4 view;
	mat4 projection;
	vec4 light;      // xyz - направление на свет, w - доля фонового света
	vec4 options;    // x - освещение вкл/выкл
} frame;

// данные объекта - у каждого объекта свой буфер и свой набор дескрипторов (набор 1)
layout(set = 1, binding = 0) uniform ObjectUniforms {
	mat4 model;
	vec4 color;      // цвет объекта
	vec4 params;     // x - умножать ли на цвета вершин
} object;

// то, что лежит в вершинном буфере (см. Vertex в geometry.hpp)
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

// это передаем дальше во фрагментный шейдер (gpu сам интерполирует между вершинами)
layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_normal;

void main() {
	// локальные координаты -> мир (model) -> камера (view) -> экран (projection)
	vec4 world = object.model * vec4(in_position, 1.0);
	gl_Position = frame.projection * frame.view * world;

	// нормаль умножаем не на model, а на обратную транспонированную к ней.
	// иначе при неравномерном растяжении нормаль перестанет быть перпендикулярна грани
	v_normal = mat3(transpose(inverse(object.model))) * in_normal;
	v_color = in_color;
}
