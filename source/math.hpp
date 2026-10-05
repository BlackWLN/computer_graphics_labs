#pragma once

// math.hpp - своя маленькая линейная алгебра: векторы, матрицы 4x4, преобразования и проекции.
// glm не подключали специально - так видно, что внутри у каждой матрицы.
//
// матрицы хранятся по столбцам (column-major), потому что так их ждет glsl:
// элемент (строка r, столбец c) лежит в m[c * 4 + r]. тогда матрицу можно просто скопировать в uniform-буфер как есть

#include <cmath>

namespace math {

constexpr float pi = 3.14159265358979323846f;

// градусы превращаем в радианы (в интерфейсе все в градусах, а sin/cos хотят радианы)
inline float radians(float degrees) { return degrees * pi / 180.0f; }

struct Vec3 {
	float x = 0.0f, y = 0.0f, z = 0.0f;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(float s, Vec3 a) { return a * s; }

// скалярное произведение
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// векторное произведение - вектор, перпендикулярный обоим
inline Vec3 cross(Vec3 a, Vec3 b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }

// делаем длину вектора равной 1
inline Vec3 normalize(Vec3 a) {
	const float len = length(a);
	return len > 1e-6f ? a * (1.0f / len) : Vec3{};
}

// матрица 4x4. at(строка, столбец) прячет внутри хранение по столбцам
struct Mat4 {
	float m[16] = {};

	float& at(int row, int col) { return m[col * 4 + row]; }
	float at(int row, int col) const { return m[col * 4 + row]; }

	static Mat4 identity() {
		Mat4 r;
		r.at(0, 0) = r.at(1, 1) = r.at(2, 2) = r.at(3, 3) = 1.0f;
		return r;
	}
};

// обычное умножение матриц: строка на столбец
inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 r;
	for (int row = 0; row < 4; ++row) {
		for (int col = 0; col < 4; ++col) {
			float sum = 0.0f;
			for (int k = 0; k < 4; ++k) {
				sum += a.at(row, k) * b.at(k, col);
			}
			r.at(row, col) = sum;
		}
	}
	return r;
}

// аффинные преобразования
// у всех последняя строка (0 0 0 1), а перенос живет в последнем столбце

// перенос на вектор t
inline Mat4 translate(Vec3 t) {
	Mat4 r = Mat4::identity();
	r.at(0, 3) = t.x;
	r.at(1, 3) = t.y;
	r.at(2, 3) = t.z;
	return r;
}

// растяжение вдоль осей
inline Mat4 scale(Vec3 s) {
	Mat4 r = Mat4::identity();
	r.at(0, 0) = s.x;
	r.at(1, 1) = s.y;
	r.at(2, 2) = s.z;
	return r;
}

// поворот вокруг оси x на угол (в радианах), дальше так же для y и z
inline Mat4 rotateX(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = Mat4::identity();
	r.at(1, 1) = c;  r.at(1, 2) = -s;
	r.at(2, 1) = s;  r.at(2, 2) = c;
	return r;
}

inline Mat4 rotateY(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = Mat4::identity();
	r.at(0, 0) = c;  r.at(0, 2) = s;
	r.at(2, 0) = -s; r.at(2, 2) = c;
	return r;
}

inline Mat4 rotateZ(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = Mat4::identity();
	r.at(0, 0) = c;  r.at(0, 1) = -s;
	r.at(1, 0) = s;  r.at(1, 1) = c;
	return r;
}

// поворот по углам эйлера (в радианах): сначала вокруг x, потом y, потом z
inline Mat4 rotateEuler(Vec3 angles) {
	return rotateZ(angles.z) * rotateY(angles.y) * rotateX(angles.x);
}

// матрица, у которой столбцы - три оси нового базиса.
// она поворачивает фигуру так, что ее оси x, y, z смотрят вдоль x_axis, y_axis, z_axis.
// используем, чтобы развернуть фигуру по направлению движения
inline Mat4 basis(Vec3 x_axis, Vec3 y_axis, Vec3 z_axis) {
	Mat4 r = Mat4::identity();
	r.at(0, 0) = x_axis.x; r.at(1, 0) = x_axis.y; r.at(2, 0) = x_axis.z;
	r.at(0, 1) = y_axis.x; r.at(1, 1) = y_axis.y; r.at(2, 1) = y_axis.z;
	r.at(0, 2) = z_axis.x; r.at(1, 2) = z_axis.y; r.at(2, 2) = z_axis.z;
	return r;
}

// камера

// матрица вида: переводит мировые координаты в координаты камеры
// (камера стоит в нуле и смотрит вдоль -z, ось y вверх).
// строим базис камеры s, u, f и "откатываем" мир на позицию камеры
inline Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
	const Vec3 f = normalize(target - eye);    // куда смотрим
	const Vec3 s = normalize(cross(f, up));    // вправо от камеры
	const Vec3 u = cross(s, f);                // настоящий верх камеры

	Mat4 r = Mat4::identity();
	r.at(0, 0) = s.x;  r.at(0, 1) = s.y;  r.at(0, 2) = s.z;
	r.at(1, 0) = u.x;  r.at(1, 1) = u.y;  r.at(1, 2) = u.z;
	r.at(2, 0) = -f.x; r.at(2, 1) = -f.y; r.at(2, 2) = -f.z;
	r.at(0, 3) = -dot(s, eye);
	r.at(1, 3) = -dot(u, eye);
	r.at(2, 3) = dot(f, eye);
	return r;
}

// проекции (доп. задание 1) 
// в vulkan все смешно
//   - ось y на экране смотрит вниз, поэтому элемент (1,1) берем с минусом
//   - глубина лежит в [0, 1], а не в [-1, 1]: ближняя плоскость -> 0, дальняя -> 1

// перспективная проекция. fovy - вертикальный угол обзора (в радианах).
// коэффициенты для z получаются из двух условий: z = -near дает глубину 0, z = -far дает 1
inline Mat4 perspective(float fovy, float aspect, float near_plane, float far_plane) {
	const float f = 1.0f / std::tan(fovy * 0.5f);

	Mat4 r;
	r.at(0, 0) = f / aspect;
	r.at(1, 1) = -f;
	r.at(2, 2) = far_plane / (near_plane - far_plane);
	r.at(2, 3) = near_plane * far_plane / (near_plane - far_plane);
	r.at(3, 2) = -1.0f; // w = -z, потом gpu делит x, y, z на w - чем дальше, тем меньше
	return r;
}

// ортографическая проекция: просто линейно растягиваем коробку
// [left, right] x [bottom, top] x [-near, -far] в куб, который видит экран. деления на w нет,
// поэтому дальние предметы не уменьшаются
inline Mat4 orthographic(float left, float right, float bottom, float top,
                         float near_plane, float far_plane) {
	Mat4 r = Mat4::identity();
	r.at(0, 0) = 2.0f / (right - left);
	r.at(1, 1) = -2.0f / (top - bottom);
	r.at(2, 2) = -1.0f / (far_plane - near_plane);
	r.at(0, 3) = -(right + left) / (right - left);
	r.at(1, 3) = (top + bottom) / (top - bottom);
	r.at(2, 3) = -near_plane / (far_plane - near_plane);
	return r;
}

} // namespace math
