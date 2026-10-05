// geometry.cpp - тут вся "чистая" геометрия без vulkan: строим вершины параллелепипеда, красим их и считаем точки траекторий.

#include "geometry.hpp"

#include <algorithm>
#include <cmath>

namespace geometry {

using math::Vec3;

// hsv -> rgb (h, s, v от 0 до 1). нужен для радуги
Vec3 hsv(float h, float s, float v) {
	const float r = std::clamp(std::fabs(h * 6.0f - 3.0f) - 1.0f, 0.0f, 1.0f);
	const float g = std::clamp(2.0f - std::fabs(h * 6.0f - 2.0f), 0.0f, 1.0f);
	const float b = std::clamp(2.0f - std::fabs(h * 6.0f - 4.0f), 0.0f, 1.0f);
	const Vec3 rgb{r, g, b};
	return (Vec3{1, 1, 1} * (1.0f - s) + rgb * s) * v;
}

namespace {
// линейная ~интерполяция~ между двумя цветами
Vec3 lerp(Vec3 a, Vec3 b, float t) { return a * (1.0f - t) + b * t; }

} // namespace

// то, что стоит под корнем в формуле объема параллелепипеда:
// V = a*b*c*sqrt(1 - cos^2(alpha) - cos^2(beta) - cos^2(gamma) + 2*cos(alpha)*cos(beta)*cos(gamma))
float volumeFactor(const ParallelepipedParams& p) {
	const float ca = std::cos(math::radians(p.alpha));
	const float cb = std::cos(math::radians(p.beta));
	const float cg = std::cos(math::radians(p.gamma));
	return 1.0f - ca * ca - cb * cb - cg * cg + 2.0f * ca * cb * cg;
}

// строим три вектора-ребра по длинам a, b, c и углам между ними
void edgeVectors(const ParallelepipedParams& p, Vec3& ea, Vec3& eb, Vec3& ec) {
	const float ca = std::cos(math::radians(p.alpha));
	const float cb = std::cos(math::radians(p.beta));
	const float cg = std::cos(math::radians(p.gamma));
	const float sg = std::sin(math::radians(p.gamma));

	// сначала единичные направления: a кладем вдоль x, b - в плоскость xy под углом gamma к a,
	// а c находим из двух условий: dot(c, a) = cos(beta) и dot(c, b) = cos(alpha).
	// третья координата c - из того, что вектор единичный (cx^2 + cy^2 + cz^2 = 1)
	const Vec3 a{1.0f, 0.0f, 0.0f};
	const Vec3 b{cg, sg, 0.0f};
	const float cy = (ca - cb * cg) / sg;
	const float cz = std::sqrt(std::max(0.0f, 1.0f - cb * cb - cy * cy));
	const Vec3 c{cb, cy, cz};

	// в сцене ось y смотрит вверх, поэтому поворачиваем все: (x, y, z) -> (x, z, -y).
	// тогда ребро c стоит "вертикально", а основание (a и b) лежит на полу
	auto toScene = [](Vec3 v) { return Vec3{v.x, v.z, -v.y}; };

	ea = toScene(a) * p.a;
	eb = toScene(b) * p.b;
	ec = toScene(c) * p.c;
}

// главная функция: вершины граней (для заливки) и ребер (для линий)
Mesh buildParallelepiped(const ParallelepipedParams& p, ColorMode mode) {
	Vec3 ea, eb, ec;
	edgeVectors(p, ea, eb, ec);

	// 8 углов: P(i, j, k) = i*a + j*b + k*c - центр, где i, j, k - это 0 или 1.
	// номер угла = i + 2j + 4k, то есть каждый бит номера отвечает за свое ребро
	const Vec3 center = (ea + eb + ec) * 0.5f;
	Vec3 corners[8];
	for (int i = 0; i < 8; ++i) {
		const float x = float(i & 1), y = float((i >> 1) & 1), z = float((i >> 2) & 1);
		corners[i] = ea * x + eb * y + ec * z - center;
	}

	// габариты фигуры - нужны, чтобы перевести координаты в цвета от 0 до 1
	Vec3 lo = corners[0], hi = corners[0];
	for (const Vec3& v : corners) {
		lo = {std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
		hi = {std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
	}
	const Vec3 size = hi - lo;

	// доп. задание 5: процедурный цвет угла в зависимости от выбранного режима
	auto colorOf = [&](int index) -> Vec3 {
		const Vec3 v = corners[index];
		switch (mode) {
		case ColorMode::PositionRGB:
			// координаты -> [0, 1] -> [0.2, 1], чтобы не было совсем черного угла
			return Vec3{(v.x - lo.x) / size.x, (v.y - lo.y) / size.y, (v.z - lo.z) / size.z} * 0.8f +
			       Vec3{0.2f, 0.2f, 0.2f};
		case ColorMode::Height:
			// от синего снизу к оранжевому сверху
			return lerp(Vec3{0.15f, 0.35f, 1.0f}, Vec3{1.0f, 0.75f, 0.2f}, (v.y - lo.y) / size.y);
		case ColorMode::CornerHue:
			// каждому из 8 углов свой оттенок
			return hsv(float(index) / 8.0f, 0.75f, 1.0f);
		case ColorMode::White:
		default:
			return {1.0f, 1.0f, 1.0f};
		}
	};

	// каждая грань - 4 угла по кругу (номера углов см. выше)
	const int faces[6][4] = {
		{0, 2, 3, 1}, // k = 0
		{4, 5, 7, 6}, // k = 1
		{0, 1, 5, 4}, // j = 0
		{2, 6, 7, 3}, // j = 1
		{0, 4, 6, 2}, // i = 0
		{1, 3, 7, 5}, // i = 1
	};

	Mesh mesh;
	mesh.triangles.reserve(36);

	for (const auto& face : faces) {
		int q[4] = {face[0], face[1], face[2], face[3]};

		// нормаль грани - через векторное произведение двух ее сторон.
		// фигура выпуклая и стоит центром в нуле, так что внешняя нормаль должна смотреть от центра.
		// если смотрит внутрь - переставляем углы. так все грани снаружи обходятся
		// против часовой стрелки, а это нужно для отсечения задних граней
		Vec3 n = math::cross(corners[q[1]] - corners[q[0]], corners[q[2]] - corners[q[0]]);
		const Vec3 face_center = (corners[q[0]] + corners[q[1]] + corners[q[2]] + corners[q[3]]) * 0.25f;
		if (math::dot(n, face_center) < 0.0f) {
			std::swap(q[1], q[3]);
			n = n * -1.0f;
		}
		n = math::normalize(n);

		// грань = 2 треугольника: (0,1,2) и (0,2,3).
		// вершины не общие между гранями, потому что у каждой грани своя нормаль
		// (поэтому вершин 6 * 6 = 36, а не 8)
		const int order[6] = {0, 1, 2, 0, 2, 3};
		for (int k : order) {
			mesh.triangles.push_back({corners[q[k]], n, colorOf(q[k])});
		}
	}

	// 12 ребер: соединяем углы, у которых номера отличаются ровно одним битом
	const Vec3 edge_color{0.05f, 0.05f, 0.07f};
	for (int i = 0; i < 8; ++i) {
		for (int bit = 1; bit < 8; bit <<= 1) {
			const int j = i | bit;
			if (j != i) {
				mesh.edges.push_back({corners[i], {}, edge_color});
				mesh.edges.push_back({corners[j], {}, edge_color});
			}
		}
	}

	return mesh;
}

// доп. задание 3: точка траектории в момент t (у всех траекторий период 2*pi)
Vec3 trajectoryPoint(const TrajectoryParams& p, float t) {
	const float R = p.radius, H = p.height;

	switch (p.type) {
	case Trajectory::Circle:
		return {R * std::cos(t), 0.0f, R * std::sin(t)};

	case Trajectory::Lissajous: {
		// фигура лиссажу - как на осциллографе, при частотах 1:2 получается восьмерка
		const float phi = math::radians(p.phase);
		return {
			R * std::sin(float(p.freq_a) * t + phi),
			H * std::sin(t),
			R * std::sin(float(p.freq_b) * t),
		};
	}

	case Trajectory::Trefoil: {
		// узел-трилистник: x = sin t + 2 sin 2t, y = cos t - 2 cos 2t, z = -sin 3t.
		// без масштаба он размером примерно 3, поэтому делим на 3 и умножаем на радиус
		const float k = R / 3.0f;
		return {
			k * (std::sin(t) + 2.0f * std::sin(2.0f * t)),
			-H * std::sin(3.0f * t),
			k * (std::cos(t) - 2.0f * std::cos(2.0f * t)),
		};
	}

	case Trajectory::Rose: {
		// полярная роза r = R*cos(k*t): при нечетном k будет k лепестков, при четном - 2k
		const float r = R * std::cos(float(p.freq_a) * t);
		return {r * std::cos(t), H * std::sin(2.0f * t), r * std::sin(t)};
	}
	}

	return {};
}

// направление движения: берем две соседние точки и вычитаем (центральная разность)
Vec3 trajectoryTangent(const TrajectoryParams& p, float t) {
	const float h = 1e-3f;
	return math::normalize(trajectoryPoint(p, t + h) - trajectoryPoint(p, t - h));
}

// разбиваем траекторию на много маленьких отрезков, чтобы нарисовать ее линией
std::vector<Vertex> trajectoryLines(const TrajectoryParams& p, Vec3 color, int segments) {
	std::vector<Vertex> lines;
	lines.reserve(size_t(segments) * 2);

	const float period = 2.0f * math::pi;
	Vec3 prev = trajectoryPoint(p, 0.0f);
	for (int i = 1; i <= segments; ++i) {
		const Vec3 cur = trajectoryPoint(p, period * float(i) / float(segments));
		lines.push_back({prev, {}, color});
		lines.push_back({cur, {}, color});
		prev = cur;
	}
	return lines;
}

} // namespace geometry
