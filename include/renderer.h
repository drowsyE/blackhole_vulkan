#pragma once

#define GLFW_INCLUDE_VULKAN
#include "include/skybox.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

static void framebufferSizeCallback(GLFWwindow *window, int width, int height);

struct Vertex {
  glm::vec2 pos;

  static VkVertexInputBindingDescription getBindingDesc() {
    VkVertexInputBindingDescription bindDesc{};
    bindDesc.binding = 0;
    bindDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindDesc.stride = sizeof(Vertex);

    return bindDesc;
  }

  static std::array<VkVertexInputAttributeDescription, 1> getAttributeDesc() {
    std::array<VkVertexInputAttributeDescription, 1> attrDescs;
    attrDescs[0] =
        (VkVertexInputAttributeDescription){.location = 0,
                                            .binding = 0,
                                            .format = VK_FORMAT_R32G32_SFLOAT,
                                            .offset = offsetof(Vertex, pos)};

    return attrDescs;
  };
};

struct GlobalUBO {
  glm::mat4 view;
  glm::mat4 proj;
};

struct CameraUBO {
  glm::mat4 viewInv;
  glm::mat4 projInv;
};

class Camera {

public:
  glm::vec3 cameraPos{0, 0, 20};
  glm::quat orientation{1, 0, 0, 0};

  float moveSpeed = 5.0f;
  float angularSpeed = glm::radians(90.0f); // deg/sec
  float currentPitch = 0.0f;

  float orbitYaw;
  float orbitPitch;
  float orbitRadius;
  char toggleOrbit = 0;

  glm::mat4 getViewMatrix() {
    // (TR)^-1 -> R^t T^-1 / R^t <-> q^-1
    glm::mat4 rot = glm::mat4_cast(glm::conjugate(orientation));
    glm::mat4 trans = glm::translate(glm::mat4(1.0f), -cameraPos);
    return rot * trans;
  }

  glm::mat4 getViewMatrixInv() {
    // TR
    glm::mat4 rot = glm::mat4_cast(orientation);
    glm::mat4 trans = glm::translate(glm::mat4(1.0f), cameraPos);
    return trans * rot;
  }

  void update(float dt, bool w, bool a, bool s, bool d, bool q, bool e,
            bool up, bool left, bool down, bool right) {

    glm::vec3 dir(0.0f);
    if (w) dir += orientation * glm::vec3(0, 0, -1);
    if (s) dir += orientation * glm::vec3(0, 0, 1);
    if (a) dir += orientation * glm::vec3(-1, 0, 0);
    if (d) dir += orientation * glm::vec3(1, 0, 0);
    if (q) dir += orientation * glm::vec3(0, -1, 0);
    if (e) dir += orientation * glm::vec3(0, 1, 0);

    if (glm::length(dir) > 0.0f)
        cameraPos += glm::normalize(dir) * moveSpeed * dt;

    float yawDelta = 0.0f;
    float pitchDelta = 0.0f;
    if (left)  yawDelta += angularSpeed * dt;
    if (right) yawDelta -= angularSpeed * dt;
    if (up)    pitchDelta += angularSpeed * dt;
    if (down)  pitchDelta -= angularSpeed * dt;

    // 3. Clamping
    float limit = glm::radians(89.0f);
    float oldPitch = currentPitch;
    currentPitch = glm::clamp(currentPitch + pitchDelta, -limit, limit);
    float actualPitchDelta = currentPitch - oldPitch;

    // 4. 쿼터니언 업데이트
    // Yaw: 세계축(0, 1, 0) 기준 왼쪽 곱셈
    if (yawDelta != 0.0f) {
        glm::quat qYaw = glm::angleAxis(yawDelta, glm::vec3(0, 1, 0));
        orientation = qYaw * orientation; 
    }

    // Pitch: 카메라 로컬 X축 기준 오른쪽 곱셈
    if (actualPitchDelta != 0.0f) {
        glm::quat qPitch = glm::angleAxis(actualPitchDelta, glm::vec3(1, 0, 0));
        orientation = orientation * qPitch;
    }

    orientation = glm::normalize(orientation);
  }

  void convertToOrbit() {
      // 1. 카메라가 현재 바라보는 방향 벡터 추출 (Forward)
      glm::vec3 forward = orientation * glm::vec3(0, 0, -1);
      
      // 2. Yaw 계산: 평면(X-Z) 상에서 -Z축과 이루는 각도
      // atan2(x, z)를 사용하여 수평 각도를 구합니다.
      orbitYaw = std::atan2(-forward.x, -forward.z);

      // 3. Pitch 계산: 수평면과 바라보는 방향 사이의 수직 각도
      // forward.y는 위아래 높이이므로, asin을 통해 각도를 구합니다.
      orbitPitch = std::asin(forward.y);

      // 4. Radius 계산 (현재 위치에서 원점까지의 거리)
      orbitRadius = glm::length(cameraPos);
  }  

  void updateOrbit(float dt, bool up, bool left, bool down, bool right, bool q, bool e) {
      // 1. 입력에 따른 각도 누적 (기존 로직) 
      if (left)  orbitYaw += angularSpeed * dt;
      if (right) orbitYaw -= angularSpeed * dt;
      if (up)    orbitPitch += angularSpeed * dt;
      if (down)  orbitPitch -= angularSpeed * dt;

      // Pitch 제한 (고개가 완전히 뒤집히지 않도록)
      orbitPitch = glm::clamp(orbitPitch, glm::radians(-89.0f), glm::radians(89.0f));

      // 2. Zoom 처리
      if (e) orbitRadius -= moveSpeed * dt;
      if (q) orbitRadius += moveSpeed * dt;
      orbitRadius = glm::max(0.1f, orbitRadius);

      // 3. Yaw와 Pitch로 새로운 쿼터니언 생성
      // 순서: 세계 Y축(Yaw) 기준 먼저 회전 후, 로컬 X축(Pitch) 회전
      glm::quat qYaw = glm::angleAxis(orbitYaw, glm::vec3(0, 1, 0));
      glm::quat qPitch = glm::angleAxis(orbitPitch, glm::vec3(1, 0, 0));
      
      orientation = qYaw * qPitch; // 핵심: 순서가 바뀌면 원점을 돌지 않습니다.
      orientation = glm::normalize(orientation);

      // 4. 위치 갱신 (원점을 중심으로 현재 방향의 반대쪽으로 배치)
      cameraPos = orientation * glm::vec3(0, 0, orbitRadius);
  }
};

class Renderer {

public:
  GLFWwindow *window;
  VkPhysicalDevice physDev;
  VkDevice device;
  VkFormat swapchainImageFormat;
  VkExtent2D swapchainImageExtent;
  VkDescriptorPool descriptorPool;
  VkRenderPass renderpass;
  bool framebufferResized = false;
  VkDescriptorSetLayout descriptorSetLayout;

  Renderer();
  ~Renderer();
  void run();
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlags,
                    VkMemoryPropertyFlags memProps, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory);
  void createImage(uint32_t width, uint32_t height, VkFormat imageFormat,
                   VkImageCreateFlags flags, uint32_t mipLevels,
                   uint32_t arrayLayers, VkImageTiling tiling,
                   VkImageUsageFlags imageUsage,
                   VkMemoryPropertyFlags properties, VkImage &image,
                   VkDeviceMemory &imageMemory);
  VkImageView createImageView(VkImage image, VkFormat format,
                              VkImageViewType viewType,
                              VkImageAspectFlags aspectFlags,
                              uint32_t mipLevels, uint32_t layerCount);
  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer commandBuffer);
  void transitionImageLayout(VkImage image, VkFormat format,
                             VkImageLayout oldLayout, VkImageLayout newLayout,
                             uint32_t mipLevels, uint32_t layerCount,
                             VkAccessFlags srcAccessMask,
                             VkAccessFlags dstAccessMask,
                             VkPipelineStageFlags srcStage,
                             VkPipelineStageFlags dstStage);

private:
  VkInstance instance;
  VkDebugUtilsMessengerEXT debugMessenger;
  VkSurfaceKHR surface;

  VkQueue graphicsQueue;
  VkQueue computeQueue;
  VkQueue presentQueue;
  uint32_t graphicsAndComputeFamilyIndex;
  uint32_t presentFamilyIndex;

  VkSwapchainKHR swapchain;
  std::vector<VkImage> swapchainImages;
  std::vector<VkImageView> swapchainImageViews;

  VkPipelineLayout graphicsPipelineLayout;
  VkPipeline graphicsPipeline;
  VkPipelineLayout computePipelineLayout;
  VkPipeline computePipeline;
  std::vector<VkFramebuffer> framebuffers;
  VkCommandPool commandPool;
  std::vector<VkCommandBuffer> commandBuffers;
  std::vector<VkCommandBuffer> computeCommandBuffers;

  std::vector<VkSemaphore> imageAvailableSemaphores;
  std::vector<VkSemaphore> renderFinishedSemaphores;
  std::vector<VkFence> inFlightFences;
  std::vector<VkSemaphore> computeFinishedSemaphores;
  std::vector<VkFence> computeInFlightFences;

  // std::vector<Vertex> vertices;
  // std::vector<uint16_t> indices;

  // VkBuffer vertexBuffer;
  // VkDeviceMemory vertexBufferMemory;

  // VkBuffer indexBuffer;
  // VkDeviceMemory indexBufferMemory;

  // GlobalUBO ubo;
  // std::vector<VkBuffer> uniformBuffers;
  // std::vector<VkDeviceMemory> uniformBuffersMemory;
  // std::vector<void *> uniformBuffersMapped;

  CameraUBO camUbo;
  std::vector<VkBuffer> cameraBuffers;
  std::vector<VkDeviceMemory> cameraBuffersMemory;
  std::vector<void *> cameraBuffersMapped;

  std::vector<VkBuffer> shaderStorageBuffers;
  std::vector<VkDeviceMemory> shaderStorageBuffersMemory;

  VkImage depthImage;
  VkImageView depthImageView;
  VkDeviceMemory depthImageMemory;

  std::vector<VkImage> outImages;
  std::vector<VkImageView> outImageViews;
  std::vector<VkDeviceMemory> outImagesMemory;


  std::vector<VkDescriptorSet> descriptorSets;
  // VkDescriptorSetLayout computeDescriptorSetLayout;
  // std::vector<VkDescriptorSet> computeDesciptorSets;

  Camera cam;
  Skybox skybox;

  void drawFrame();

  void initWindow();
  void createInstance();
  void setupDebugMessenger();
  void createSurface();
  void selectPhysicalDevice();
  void createLogicalDevice();
  void createSwapchain();
  void createImageViews();
  void createRenderpass();
  void createGraphicsPipeline();
  void createComputePipeline();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createComputeCommandBuffers();
  void createSyncObjects();
  void recordCommandbuffer(VkCommandBuffer &commandBuffer, uint32_t imageIndex);
  void recordComputeCommandbuffer(VkCommandBuffer &commandbuffer);
  void createVertexBuffer(std::vector<Vertex> &vertices);
  void createIndexBuffer(std::vector<uint16_t> &indices);
  template <typename T>
  void createUniformBuffers(size_t count, std::vector<VkBuffer> &buffers, std::vector<VkDeviceMemory> &buffersMemory, std::vector<void*> &pData);
  void updateUniformBuffer(uint32_t currentFrame, bool resized);
  // void createShaderStorageBuffers();
  void createDescriptorPool();
  void createDescriptorSetLayout();
  void createDescriptorSets();
  void createDepthResources();
  void createOutputImages();

  void createSkybox();
  void processInput1(GLFWwindow* window, Camera& cam, float dt);
  void processInput2(GLFWwindow* window, Camera& cam, float dt);
  void (Renderer::*fptr)(GLFWwindow *, Camera &, float);
  void (Renderer::*inputs[2])(GLFWwindow*, Camera&, float) = {&Renderer::processInput1, &Renderer::processInput2};

  void cleanupSwapchain();
  void recreateSwapchain(uint32_t imageIndex);
  void clearUBO();
  void clearCameraUBO();
  void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties);
  VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features);
  VkFormat findDepthFormat();
};
