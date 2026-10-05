#pragma once

// geometry.hpp - объявления всего, что связано с формой фигуры и траекториями. тут нет ничего про vulkan, только математика и массивы вершин

#include <cstdint>
#include <vector>

#include "math.hpp"

namespace geometry {

// одна вершина. порядок полей совпадает с входами вершинного шейдера (location 0, 1, 2)
struct Vertex {
	math::Vec3 position; // координаты в локальной системе фигуры
	math::Vec3 normal;   // нормаль грани, нужна для освещения
	math::Vec3 color;    // процедурный цвет вершины
};

// параллелепипед задаем длинами ребер a, b, c и углами между ними
// (как ячейку кристаллической решетки):
//   alpha - угол между ребрами b и c
//   beta  - угол между ребрами a и c
//   gamma - угол между ребрами a и b
// если все углы по 90 градусов - получится обычный прямоугольный параллелепипед
struct ParallelepipedParams {
	float a = 2.0f;
	float b = 1.3f;
	float c = 1.0f;
	float alpha = 80.0f; // градусы
	float beta = 70.0f;
	float gamma = 65.0f;
};

// как красить вершины (доп. задание 5)
enum class ColorMode : int {
	PositionRGB = 0, // координаты (x, y, z) -> цвет (r, g, b)
	Height = 1,      // градиент по высоте
	CornerHue = 2,   // у каждого из 8 углов свой оттенок
	White = 3,       // без окраски
};

struct Mesh {
	std::vector<Vertex> triangles; // треугольники граней (по 3 вершины)
	std::vector<Vertex> edges;     // отрезки ребер (по 2 вершины)
};

// объем = a*b*c*sqrt(volumeFactor). если volumeFactor <= 0, таких углов не бывает
float volumeFactor(const ParallelepipedParams& p);

// три вектора-ребра, которые выходят из одного угла
void edgeVectors(const ParallelepipedParams& p, math::Vec3& ea, math::Vec3& eb, math::Vec3& ec);

// вершины граней и ребер параллелепипеда
Mesh buildParallelepiped(const ParallelepipedParams& p, ColorMode mode);

// цвет из hsv (оттенок, насыщенность, яркость от 0 до 1) в rgb
math::Vec3 hsv(float h, float s, float v);

//траектории (доп. задание 3)
// все траектории замкнутые и повторяются через t = 2*pi

enum class Trajectory : int {
	Circle = 0,     // окружность
	Lissajous = 1,  // фигура лиссажу (восьмерка и т.п.) + покачивание вверх-вниз
	Trefoil = 2,    // узел-трилистник
	Rose = 3,       // роза r = R*cos(k*t) + покачивание вверх-вниз
};

struct TrajectoryParams {
	Trajectory type = Trajectory::Lissajous;
	float radius = 2.5f;     // размер траектории
	float height = 0.8f;     // насколько качается по высоте
	int freq_a = 1;          // частоты лиссажу (целые, чтобы кривая замыкалась); для розы freq_a - это k
	int freq_b = 2;
	float phase = 90.0f;     // сдвиг фаз лиссажу, градусы
};

// точка траектории в момент t
math::Vec3 trajectoryPoint(const TrajectoryParams& p, float t);

// куда летим в момент t (единичный вектор по касательной)
math::Vec3 trajectoryTangent(const TrajectoryParams& p, float t);

// вся траектория в виде отрезков, чтобы нарисовать ее линией
std::vector<Vertex> trajectoryLines(const TrajectoryParams& p, math::Vec3 color, int segments = 400);

} // namespace geometry
