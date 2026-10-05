#version 450

// фрагментный шейдер: запускается для каждого пикселя фигуры и решает, какого он цвета

layout(set = 0, binding = 0) uniform FrameUniforms {
	mat4 view;
	mat4 projection;
	vec4 light;
	vec4 options;
} frame;

layout(set = 1, binding = 0) uniform ObjectUniforms {
	mat4 model;
	vec4 color;
	vec4 params;
} object;

// push-константа: x - что рисуем (0 - грани, 1 - линии)
layout(push_constant) uniform DrawConstants {
	vec4 params;
} draw;

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 out_color;

void main() {
	// линии (ребра, траектория, нити) просто красим их цветом, без освещения
	if (draw.params.x > 0.5) {
		out_color = vec4(v_color, 1.0);
		return;
	}

	// доп. задания 4 и 5: цвет из интерфейса умножаем на процедурный цвет вершины.
	// mix(1, цвет, 0) = 1, то есть если умножать не надо - умножаем на белый
	vec3 vertex_color = mix(vec3(1.0), v_color, object.params.x);
	vec3 base = object.color.rgb * vertex_color;

	if (frame.options.x > 0.5) {
		// освещение по ламберту: чем прямее свет падает на грань, тем она ярче.
		// плюс фоновый свет, чтобы теневые грани не были совсем черными
		float ambient = frame.light.w;
		float diffuse = max(dot(normalize(v_normal), normalize(frame.light.xyz)), 0.0);
		base *= ambient + (1.0 - ambient) * diffuse;
	}

	out_color = vec4(base, 1.0);
}
