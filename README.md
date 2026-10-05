# Лабораторная работа №1 — Основы 3D-графики

## Условие

**Вариант 3 — параллелепипед.**

Нужно построить 3D-объект и спроецировать его на экран, поработать с матрицами перспективной и ортографической проекции и с аффинными преобразованиями. Программа должна работать в реальном времени, с возможностью менять проекцию и преобразования объекта. Используются Vulkan, GLFW и ImGui.

Выполнены все дополнительные задания:

1. Переключение проекции (ортографическая / перспективная) в интерфейсе.
2. Изменение позиции, поворота и растяжения фигуры.
3. Движение и вращение по сложной траектории с паузой, скоростью и параметрами траектории.
4. Выбор цвета фигуры (`ColorEdit`).
5. Процедурные цвета вершин, которые умножаются на выбранный цвет.
6. Несколько объектов, у каждого свой `VkDescriptorSet`: сцена «Маятниковая волна» (переключатель вверху панели).

## Что нужно установить

- C++20-компилятор (clang, gcc или MSVC)
- CMake 3.20+ и Ninja
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) версии **1.4.321 или новее**

На macOS:
```bash
xcode-select --install
brew install cmake ninja
# Vulkan SDK ставится установщиком с сайта LunarG (с галочкой "System Global Installation"),
# затем в ~/.zshrc добавить:
source ~/VulkanSDK/<версия>/setup-env.sh
```

GLFW, ImGui, vk-bootstrap и VMA скачиваются автоматически при первой сборке, нужен интернет.

## Сборка и запуск

```bash
git clone https://github.com/BlackWLN/computer_graphics_labs.git
cd computer_graphics_labs
git switch lab1

cmake --preset debug
cmake --build build-debug --parallel
./build-debug/vulkan-starter-app
```

На Windows с Visual Studio: `cmake --preset msvc-debug`, затем `cmake --build build-debug`. Исполняемый файл будет в `build-debug/Debug/`.

## Управление

- Левая кнопка мыши по сцене вращает камеру, колесо приближает.
- Все параметры настраиваются в панели «Параметры».

Проект основан на [стартовом коде преподавателя](https://github.com/vladeemerr/vulkan-starter-app).
