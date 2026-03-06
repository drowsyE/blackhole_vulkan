// #define NDEBUG

#include "include/renderer.h"
#include "include/skybox.h"
#include "include/utils.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <stdio.h>
#include <vector>

#define CUBEMAP_TEXTURE_PATH "textures/cubemap_images/"

#define DEFAULT_WIDTH 800
#define DEFAULT_HEIGHT 600
#define WINDOW_TITLE "Black hole simulation"

#define MAX_FRAME_IN_FLIGHT 2

int currentFrame = 0;

#ifdef NDEBUG
bool enableValidationLayers = false;
#else
bool enableValidationLayers = true;
#endif

std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,

#ifdef __APPLE__
                                              "VK_KHR_portability_subset"
#endif
};

Renderer::Renderer() {
  initWindow();
  createInstance();
  setupDebugMessenger();
  createSurface();
  selectPhysicalDevice();
  createLogicalDevice();
  createSwapchain();
  createImageViews();
  createDepthResources();

  createCommandPool();
  createCommandBuffers();
  createRenderpass();
  createFramebuffers();

  createDescriptorSetLayout();
  createDescriptorPool();
  // ----- descriptors between here ------

  // createUniformBuffers<GlobalUBO>(MAX_FRAME_IN_FLIGHT, uniformBuffers,
  //                                 uniformBuffersMemory,
  //                                 uniformBuffersMapped);
  createUniformBuffers<CameraUBO>(MAX_FRAME_IN_FLIGHT, cameraBuffers,
                                  cameraBuffersMemory, cameraBuffersMapped);

  createOutputImages();
  createSkybox(); // -> uses renderer's descriptor pool
  // ----- and here ------
  createDescriptorSets();

  // createGraphicsPipeline();
  createComputePipeline();
  createComputeCommandBuffers();
  createSyncObjects();

  // clearUBO();
  clearCameraUBO();

  // createShaderStorageBuffers();
}

Renderer::~Renderer() {

  vkDeviceWaitIdle(device);

  skybox.destroy();

  // // destroy shader storage buffer
  // for (VkDeviceMemory &ssbMem : shaderStorageBuffersMemory) {
  //     vkFreeMemory(device, ssbMem, nullptr);
  // }
  // for (VkBuffer &ssb : shaderStorageBuffers) {
  //     vkDestroyBuffer(device, ssb, nullptr);
  // }

  for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
    vkDestroyImageView(device, outImageViews[i], nullptr);
    vkDestroyImage(device, outImages[i], nullptr);
    vkFreeMemory(device, outImagesMemory[i], nullptr);
  }
  vkDestroyImage(device, depthImage, nullptr);
  vkDestroyImageView(device, depthImageView, nullptr);
  vkFreeMemory(device, depthImageMemory, nullptr);

  for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
    // vkDestroyBuffer(device, uniformBuffers[i], nullptr);
    vkDestroyBuffer(device, cameraBuffers[i], nullptr);

    // vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    vkFreeMemory(device, cameraBuffersMemory[i], nullptr);
  }

  for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
    vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
    vkDestroyFence(device, inFlightFences[i], nullptr);
    vkDestroySemaphore(device, computeFinishedSemaphores[i], nullptr);
    vkDestroyFence(device, computeInFlightFences[i], nullptr);
  }
  for (int i = 0; i < swapchainImages.size(); i++) {
    vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
  }
  vkDestroyCommandPool(device, commandPool, nullptr);
  for (VkFramebuffer &framebuffer : framebuffers) {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }
  vkDestroyPipeline(device, computePipeline, nullptr);
  // vkDestroyPipeline(device, graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
  // vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);
  vkDestroyDescriptorPool(device, descriptorPool, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
  // vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);
  vkDestroyRenderPass(device, renderpass, nullptr);
  for (VkImageView &view : swapchainImageViews) {
    vkDestroyImageView(device, view, nullptr);
  }
  vkDestroySwapchainKHR(device, swapchain, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroySurfaceKHR(instance, surface, nullptr);
  if (enableValidationLayers) {
    destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
  }
  vkDestroyInstance(instance, nullptr);
  glfwDestroyWindow(window);
  glfwTerminate();
}

void Renderer::run() {
  void (*fptr)(GLFWwindow *, Camera &, float);
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    processInput1(window, cam, 0.01);
    drawFrame();
  }

  vkDeviceWaitIdle(device);
}

void Renderer::drawFrame() {
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  // compute submission
  vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE,
                  UINT64_MAX);

  // updateUniformBuffer(currentFrame);
  vkResetFences(device, 1, &computeInFlightFences[currentFrame]);
  vkResetCommandBuffer(computeCommandBuffers[currentFrame], 0);

  recordComputeCommandbuffer(computeCommandBuffers[currentFrame]);

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];
  vkQueueSubmit(computeQueue, 1, &submitInfo,
                computeInFlightFences[currentFrame]);

  // graphics submission
  vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
                  UINT64_MAX);

  uint32_t imageIndex;
  VkResult res = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                       imageAvailableSemaphores[currentFrame],
                                       VK_NULL_HANDLE, &imageIndex);
  if (res == VK_ERROR_OUT_OF_DATE_KHR || framebufferResized) {
    framebufferResized = false;
    recreateSwapchain(-1);
    return;
  } else if (res != VK_SUBOPTIMAL_KHR && res != VK_SUCCESS) {
    throw std::runtime_error("Failed to acquired swapchain image!");
  }

  vkResetFences(device, 1, &inFlightFences[currentFrame]);

  vkResetCommandBuffer(commandBuffers[currentFrame], 0);

  recordCommandbuffer(commandBuffers[currentFrame], imageIndex);
  VkSemaphore waitSemaphores[] = {computeFinishedSemaphores[currentFrame],
                                  imageAvailableSemaphores[currentFrame]};
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

  updateUniformBuffer(currentFrame, false);

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
  submitInfo.waitSemaphoreCount = 2;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &renderFinishedSemaphores[imageIndex];

  res = vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                      inFlightFences[currentFrame]);
  if (res != VK_SUCCESS) {
    throw std::runtime_error("Failed at vkQueueSubmit!");
  }

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.pImageIndices = &imageIndex;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swapchain;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinishedSemaphores[imageIndex];
  res = vkQueuePresentKHR(presentQueue, &presentInfo);

  currentFrame = (currentFrame + 1) % MAX_FRAME_IN_FLIGHT;
}

void Renderer::initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window = glfwCreateWindow(DEFAULT_WIDTH, DEFAULT_HEIGHT, WINDOW_TITLE,
                            nullptr, nullptr);

  glfwSetWindowUserPointer(window, this);
  glfwSetWindowSizeCallback(window, framebufferSizeCallback);
}

void Renderer::createInstance() {

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.apiVersion = VK_API_VERSION_1_2;
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pApplicationName = WINDOW_TITLE;
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No engine";

  uint32_t count;
  const char **exts;
  exts = glfwGetRequiredInstanceExtensions(&count);
  std::vector<const char *> instanceExtensions(exts, exts + count);

#ifdef __APPLE__
  instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  instanceExtensions.push_back(
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

  if (enableValidationLayers) {
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  VkDebugUtilsMessengerCreateInfoEXT messengerCI{};
  populateDebugMessenger(messengerCI);

  VkInstanceCreateInfo instanceInfo{};
  instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceInfo.pApplicationInfo = &appInfo;
  instanceInfo.enabledExtensionCount = instanceExtensions.size();
  instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();

  if (enableValidationLayers) {
    instanceInfo.enabledLayerCount = validationLayers.size();
    instanceInfo.ppEnabledLayerNames = validationLayers.data();
    instanceInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&messengerCI;
  } else {
    instanceInfo.enabledLayerCount = 0;
    instanceInfo.ppEnabledLayerNames = nullptr;
  }

#ifdef __APPLE__
  instanceInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

  chk(vkCreateInstance(&instanceInfo, nullptr, &instance), "vkCreateInstance");
}

void Renderer::setupDebugMessenger() {
  if (!enableValidationLayers)
    return;

  VkDebugUtilsMessengerCreateInfoEXT createInfo{};
  populateDebugMessenger(createInfo);

  chk(createDebugUtilsMessenger(instance, &createInfo, nullptr,
                                &debugMessenger),
      "createDebugUtilsMessenger");
}

void Renderer::createSurface() {
  chk(glfwCreateWindowSurface(instance, window, nullptr, &surface),
      "glfwCreateWindowSurface");
}

void Renderer::selectPhysicalDevice() {
  uint32_t devCnt;
  vkEnumeratePhysicalDevices(instance, &devCnt, nullptr);
  std::vector<VkPhysicalDevice> devices(devCnt);
  vkEnumeratePhysicalDevices(instance, &devCnt, devices.data());

  for (const VkPhysicalDevice &device : devices) {
    VkPhysicalDeviceProperties devProps{};
    vkGetPhysicalDeviceProperties(device, &devProps);

    uint32_t qPropsCnt;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &qPropsCnt, nullptr);
    std::vector<VkQueueFamilyProperties> qFamilyProps(qPropsCnt);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &qPropsCnt,
                                             qFamilyProps.data());

    graphicsAndComputeFamilyIndex = -1;
    presentFamilyIndex = -1;
    int i = 0;
    for (const VkQueueFamilyProperties &props : qFamilyProps) {
      if (qFamilyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT &&
          qFamilyProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        graphicsAndComputeFamilyIndex = i;
      }

      VkBool32 presentSupport = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
      if (presentSupport) {
        presentFamilyIndex = i;
      }

      if (graphicsAndComputeFamilyIndex != -1 && presentFamilyIndex != -1) {
        physDev = device;
        printf("\n[Info] | Device selected : %s\n", devProps.deviceName);
        printf("[Info] | Graphics Family : %d, Present Family : %d\n",
               graphicsAndComputeFamilyIndex, presentFamilyIndex);
        return;
      }

      ++i;
    }
  }
  throw std::runtime_error(
      "There is no available physical device supporting vulkan!");
}

void Renderer::createLogicalDevice() {

  std::vector<VkDeviceQueueCreateInfo> qCIs;
  float priorities = 1.0f;

  VkDeviceQueueCreateInfo qInfo{};
  qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qInfo.queueFamilyIndex = graphicsAndComputeFamilyIndex;
  qInfo.queueCount = 1;
  qInfo.pQueuePriorities = &priorities;
  qCIs.push_back(qInfo);

  if (graphicsAndComputeFamilyIndex != presentFamilyIndex) {
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = presentFamilyIndex;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &priorities;
    qCIs.push_back(qInfo);
  }

  VkPhysicalDeviceFeatures supportedFeatures{};
  vkGetPhysicalDeviceFeatures(physDev, &supportedFeatures);

  VkPhysicalDeviceFeatures deviceFeatures{};
  if (supportedFeatures.samplerAnisotropy) {
    deviceFeatures.samplerAnisotropy = VK_TRUE; // enable anisotrophic filtering
  }

  VkDeviceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  info.queueCreateInfoCount = qCIs.size();
  info.pQueueCreateInfos = qCIs.data();
  info.enabledExtensionCount = deviceExtensions.size();
  info.ppEnabledExtensionNames = deviceExtensions.data();
  info.pEnabledFeatures = &deviceFeatures;
  info.enabledExtensionCount = deviceExtensions.size();
  info.ppEnabledExtensionNames = deviceExtensions.data();
  if (enableValidationLayers) {
    info.enabledLayerCount = validationLayers.size();
    info.ppEnabledLayerNames = validationLayers.data();
  } else {
    info.enabledLayerCount = 0;
    info.ppEnabledLayerNames = nullptr;
  }

  chk(vkCreateDevice(physDev, &info, nullptr, &device), "vkCreateDevice");

  vkGetDeviceQueue(device, graphicsAndComputeFamilyIndex, 0, &graphicsQueue);
  vkGetDeviceQueue(device, graphicsAndComputeFamilyIndex, 0, &computeQueue);
  vkGetDeviceQueue(device, presentFamilyIndex, 0, &presentQueue);
}

void Renderer::createSwapchain() {

  // min image count
  VkSurfaceCapabilitiesKHR surfaceCaps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDev, surface, &surfaceCaps);
  uint32_t minImgs;
  if (surfaceCaps.minImageCount + 1 > surfaceCaps.maxImageCount) {
    minImgs = surfaceCaps.maxImageCount;
  } else {
    minImgs = surfaceCaps.minImageCount + 1;
  }

  uint32_t fmtCnt;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, surface, &fmtCnt, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(fmtCnt);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, surface, &fmtCnt,
                                       formats.data());

  // image format & colorspace
  VkFormat imageFormat;
  VkColorSpaceKHR imageColorSpace;
  for (const VkSurfaceFormatKHR &fmt : formats) {
    if (fmt.format == VK_FORMAT_R8G8B8_SRGB &&
        fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      imageFormat = fmt.format;
      imageColorSpace = fmt.colorSpace;
      break;
    }
  }
  if (imageFormat != VK_FORMAT_R8G8B8_SRGB) {
    imageFormat = formats[0].format;
    imageColorSpace = formats[0].colorSpace;
  }
  swapchainImageFormat = imageFormat;

  // image extent
  VkExtent2D imageExtent;
  if (surfaceCaps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    imageExtent = surfaceCaps.currentExtent;
  } else {
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    imageExtent.width = std::clamp<int>(width, surfaceCaps.minImageExtent.width,
                                        surfaceCaps.maxImageExtent.width);
    imageExtent.height =
        std::clamp<int>(height, surfaceCaps.minImageExtent.height,
                        surfaceCaps.maxImageExtent.height);
  }
  swapchainImageExtent = imageExtent;

  // image sharing mode
  VkSharingMode imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  std::vector<uint32_t> queueFamilyIndices = {graphicsAndComputeFamilyIndex};
  if (graphicsAndComputeFamilyIndex != presentFamilyIndex) {
    imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    queueFamilyIndices.push_back(presentFamilyIndex);
  }

  // present mode
  VkPresentModeKHR presentMode;
  uint32_t modeCnt;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, surface, &modeCnt,
                                            nullptr);
  std::vector<VkPresentModeKHR> presentModes(modeCnt);
  vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, surface, &modeCnt,
                                            presentModes.data());
  for (const VkPresentModeKHR &mode : presentModes) {
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
      presentMode = mode;
    }
    break;
  }
  if (presentMode != VK_PRESENT_MODE_MAILBOX_KHR) {
    presentMode = VK_PRESENT_MODE_FIFO_KHR;
  }

  VkSwapchainCreateInfoKHR swapInfo{
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .pNext = nullptr,
      .flags = 0,
      .surface = surface,
      .minImageCount = minImgs,
      .imageFormat = imageFormat,
      .imageColorSpace = imageColorSpace,
      .imageExtent = imageExtent,
      .imageArrayLayers = 1,
      .imageUsage =
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .imageSharingMode = imageSharingMode,
      .queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size()),
      .pQueueFamilyIndices = queueFamilyIndices.data(),
      .preTransform = surfaceCaps.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = presentMode,
      .clipped = VK_TRUE,
      .oldSwapchain = VK_NULL_HANDLE};

  chk(vkCreateSwapchainKHR(device, &swapInfo, nullptr, &swapchain),
      "vkCreateSwapchainKHR");

  vkGetSwapchainImagesKHR(device, swapchain, &minImgs, nullptr);
  swapchainImages.resize(minImgs);
  vkGetSwapchainImagesKHR(device, swapchain, &minImgs, swapchainImages.data());
}

void Renderer::createImageViews() {
  swapchainImageViews.resize(swapchainImages.size());

  for (int i = 0; i < swapchainImages.size(); i++) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = swapchainImages[i];
    viewInfo.format = swapchainImageFormat;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    // swapchain image view's components must be SWIZZLE_IDENTITY
    viewInfo.components = {.r = VK_COMPONENT_SWIZZLE_IDENTITY,
                           .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                           .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                           .a = VK_COMPONENT_SWIZZLE_IDENTITY};
    viewInfo.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1};

    chk(vkCreateImageView(device, &viewInfo, nullptr, &swapchainImageViews[i]),
        "vkCreateImageView");
  }
}

void Renderer::createRenderpass() {

  // attachments
  std::vector<VkAttachmentDescription> attachments;
  VkAttachmentDescription colorAtt{};
  colorAtt.format = swapchainImageFormat;
  colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  attachments.push_back(colorAtt);

  VkAttachmentDescription depthAtt{};
  depthAtt.format = findDepthFormat();
  depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  attachments.push_back(depthAtt);

  VkAttachmentReference colorRef{};
  colorRef.attachment = 0;
  colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthRef{};
  depthRef.attachment = 1;
  depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  // end of attachments

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorRef;
  subpass.pDepthStencilAttachment = &depthRef;

  // external <-> subpass 0 dependency
  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  createInfo.attachmentCount = attachments.size();
  createInfo.pAttachments = attachments.data();
  createInfo.subpassCount = 1;
  createInfo.pSubpasses = &subpass;
  createInfo.dependencyCount = 1;
  createInfo.pDependencies = &dependency;

  chk(vkCreateRenderPass(device, &createInfo, nullptr, &renderpass),
      "vkCreateRenderPass");
}

void Renderer::createGraphicsPipeline() {

  // ----- vertex input -------
  auto bindingDesc = Vertex::getBindingDesc();
  auto attributeDesc = Vertex::getAttributeDesc();

  VkPipelineVertexInputStateCreateInfo inputInfo{};
  inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  inputInfo.vertexBindingDescriptionCount = 1;
  inputInfo.pVertexBindingDescriptions = &bindingDesc;
  inputInfo.vertexAttributeDescriptionCount = attributeDesc.size();
  inputInfo.pVertexAttributeDescriptions = attributeDesc.data();
  // ------ end of vertex input -----

  // -------- Input assembly ------------
  VkPipelineInputAssemblyStateCreateInfo assemInfo{};
  assemInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  assemInfo.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  assemInfo.primitiveRestartEnable = VK_FALSE;
  // ------- end of input assembly -------

  // ----------- shader -----------
  VkShaderModule vertexShaderModule =
      createShader(device, "shaders/renderer/spv/vert.spv");
  VkShaderModule fragShaderModule =
      createShader(device, "shaders/renderer/spv/frag.spv");

  VkPipelineShaderStageCreateInfo vertexShaderCI{};
  vertexShaderCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertexShaderCI.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertexShaderCI.module = vertexShaderModule;
  vertexShaderCI.pName = "main";

  VkPipelineShaderStageCreateInfo fragShaderCI{};
  fragShaderCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderCI.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderCI.module = fragShaderModule;
  fragShaderCI.pName = "main";

  VkPipelineShaderStageCreateInfo stageInfos[] = {vertexShaderCI, fragShaderCI};
  // ------- end of shader -------

  // skipping tessellation / geometry shader

  // ---------- Rasterizer ----------
  VkPipelineRasterizationStateCreateInfo rasterInfo{};
  rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterInfo.depthClampEnable = VK_FALSE;
  rasterInfo.rasterizerDiscardEnable = VK_FALSE;
  rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
  rasterInfo.lineWidth = 1.0f;
  rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterInfo.depthBiasEnable = VK_FALSE;
  rasterInfo.depthBiasConstantFactor = 0.0f;
  rasterInfo.depthBiasClamp = 0.0f;
  rasterInfo.depthBiasSlopeFactor = 0.0f;
  // ------- end of rasterizer -------

  // --------- Color blending ---------
  VkPipelineColorBlendAttachmentState blendAttachment{};
  blendAttachment.blendEnable = VK_FALSE;
  blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
  blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
  blendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  VkPipelineColorBlendStateCreateInfo blendInfo{};
  blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blendInfo.logicOpEnable = VK_FALSE;
  blendInfo.logicOp = VK_LOGIC_OP_COPY;
  blendInfo.attachmentCount = 1;
  blendInfo.pAttachments = &blendAttachment;
  blendInfo.blendConstants[0] = 0.0f;
  blendInfo.blendConstants[1] = 0.0f;
  blendInfo.blendConstants[2] = 0.0f;
  blendInfo.blendConstants[3] = 0.0f;
  // ----- end of color blending -----

  // ---------- pipeline layout --------- //
  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = 0;
  layoutInfo.pSetLayouts = nullptr;
  layoutInfo.pushConstantRangeCount = 0;
  layoutInfo.pPushConstantRanges = nullptr;
  vkCreatePipelineLayout(device, &layoutInfo, nullptr, &graphicsPipelineLayout);
  // --------- end of pipeline layout ------

  // --------- viewport / scissor ---------
  VkPipelineViewportStateCreateInfo viewportInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .viewportCount = 1,
      .pViewports = nullptr, // for dynamic state
      .scissorCount = 1,
      .pScissors = nullptr, // for dynamic state
  };
  // ------- end of viewport / scissors ----

  // ----- dynamic state -----
  VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                     VK_DYNAMIC_STATE_SCISSOR};

  VkPipelineDynamicStateCreateInfo dynamicStateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamicStates};
  // ----- end of dynamic state -----

  // ----- multisampling -----
  VkPipelineMultisampleStateCreateInfo sampleInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = VK_FALSE,
      .minSampleShading = 1.0f,
      .pSampleMask = nullptr,
      .alphaToCoverageEnable = VK_FALSE,
      .alphaToOneEnable = VK_FALSE};
  // ----- end of multisampling ----

  VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .stageCount = 2,
      .pStages = stageInfos,
      .pVertexInputState = &inputInfo,
      .pInputAssemblyState = &assemInfo,
      .pTessellationState = VK_NULL_HANDLE,
      .pViewportState = &viewportInfo,
      .pRasterizationState = &rasterInfo,
      .pMultisampleState = &sampleInfo,
      .pDepthStencilState = VK_NULL_HANDLE,
      .pColorBlendState = &blendInfo,
      .pDynamicState = &dynamicStateInfo,
      .layout = graphicsPipelineLayout,
      .renderPass = renderpass,
      .subpass = 0,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = -1};

  chk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                &graphicsPipelineCreateInfo, nullptr,
                                &graphicsPipeline),
      "vkCreateGraphicsPipelines");

  vkDestroyShaderModule(device, vertexShaderModule, nullptr);
  vkDestroyShaderModule(device, fragShaderModule, nullptr);
}

void Renderer::createComputePipeline() {

  VkShaderModule computeShaderModule =
      createShader(device, "shaders/renderer/spv/comp.spv");

  VkPipelineShaderStageCreateInfo computeShaderCI{};
  computeShaderCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  computeShaderCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  computeShaderCI.module = computeShaderModule;
  computeShaderCI.pName = "main";

  VkDescriptorSetLayout setLayouts[] = {descriptorSetLayout,
                                        skybox.skyboxDescriptorSetLayout};

  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(float);

  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges = &pushConstantRange;
  layoutInfo.setLayoutCount = 2;
  layoutInfo.pSetLayouts = setLayouts;
  vkCreatePipelineLayout(device, &layoutInfo, nullptr, &computePipelineLayout);

  VkComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.layout = computePipelineLayout;
  pipelineInfo.stage = computeShaderCI;
  vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                           &computePipeline);

  vkDestroyShaderModule(device, computeShaderModule, nullptr);
}

void Renderer::createFramebuffers() {
  framebuffers.resize(swapchainImages.size());

  for (int i = 0; i < swapchainImages.size(); i++) {
    VkImageView attachments[] = {swapchainImageViews[i], depthImageView};

    VkFramebufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    info.width = swapchainImageExtent.width;
    info.height = swapchainImageExtent.height;
    info.attachmentCount = 2;
    info.pAttachments = attachments;
    info.renderPass = renderpass;
    info.layers = 1;

    chk(vkCreateFramebuffer(device, &info, nullptr, &framebuffers[i]),
        "vkCreateFramebuffer");
  }
}

void Renderer::createCommandPool() {
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = graphicsAndComputeFamilyIndex;
  chk(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool),
      "vkCreateCommandPool");
}

void Renderer::createCommandBuffers() {
  commandBuffers.resize(MAX_FRAME_IN_FLIGHT);

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = commandBuffers.size();
  allocInfo.commandPool = commandPool;

  chk(vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()),
      "vkAllocateCommandBuffers");
}

void Renderer::createComputeCommandBuffers() {
  computeCommandBuffers.resize(MAX_FRAME_IN_FLIGHT);

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = computeCommandBuffers.size();
  allocInfo.commandPool = commandPool;

  chk(vkAllocateCommandBuffers(device, &allocInfo,
                               computeCommandBuffers.data()),
      "vkAllocateCommandBuffers");
}

void Renderer::createSyncObjects() {

  imageAvailableSemaphores.resize(MAX_FRAME_IN_FLIGHT);
  renderFinishedSemaphores.resize(swapchainImages.size());
  inFlightFences.resize(MAX_FRAME_IN_FLIGHT);
  computeFinishedSemaphores.resize(MAX_FRAME_IN_FLIGHT);
  computeInFlightFences.resize(MAX_FRAME_IN_FLIGHT);

  VkSemaphoreCreateInfo semaInfo{};
  semaInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaInfo.flags = VK_SEMAPHORE_TYPE_BINARY;

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
    vkCreateSemaphore(device, &semaInfo, nullptr, &imageAvailableSemaphores[i]);
    vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]);

    vkCreateSemaphore(device, &semaInfo, nullptr,
                      &computeFinishedSemaphores[i]);
    vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFences[i]);
  }
  for (int i = 0; i < swapchainImages.size(); i++) {
    chk(vkCreateSemaphore(device, &semaInfo, nullptr,
                          &renderFinishedSemaphores[i]),
        "vkCreateSemaphore");
  }
}

void Renderer::recordCommandbuffer(VkCommandBuffer &commandBuffer,
                                   uint32_t imageIndex) {

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  VkImageSubresourceRange subresourceRange = {.aspectMask =
                                                  VK_IMAGE_ASPECT_COLOR_BIT,
                                              .baseMipLevel = 0,
                                              .levelCount = 1,
                                              .baseArrayLayer = 0,
                                              .layerCount = 1};

  // === 1. outImage: GENERAL -> TRANSFER_SRC_OPTIMAL ===
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.image = outImages[currentFrame];
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.subresourceRange = subresourceRange;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  // === 2. swapchain image: UNDEFINED -> TRANSFER_DST_OPTIMAL ===
  VkImageMemoryBarrier swapBarrier{};
  swapBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  swapBarrier.image = swapchainImages[imageIndex];
  swapBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  swapBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  swapBarrier.subresourceRange = subresourceRange;
  swapBarrier.srcAccessMask = 0;
  swapBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  swapBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  swapBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &swapBarrier);

  // === 3. Blit outImage -> swapchain image ===
  // Using vkCmdBlitImage to handle potential format differences
  // (outImage: R8G8B8A8_UNORM, swapchain: may differ)
  VkImageBlit blitRegion{};
  blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blitRegion.srcSubresource.mipLevel = 0;
  blitRegion.srcSubresource.baseArrayLayer = 0;
  blitRegion.srcSubresource.layerCount = 1;
  blitRegion.srcOffsets[0] = {0, 0, 0};
  blitRegion.srcOffsets[1] = {static_cast<int32_t>(swapchainImageExtent.width),
                              static_cast<int32_t>(swapchainImageExtent.height),
                              1};
  blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blitRegion.dstSubresource.mipLevel = 0;
  blitRegion.dstSubresource.baseArrayLayer = 0;
  blitRegion.dstSubresource.layerCount = 1;
  blitRegion.dstOffsets[0] = {0, 0, 0};
  blitRegion.dstOffsets[1] = {static_cast<int32_t>(swapchainImageExtent.width),
                              static_cast<int32_t>(swapchainImageExtent.height),
                              1};

  vkCmdBlitImage(
      commandBuffer, outImages[currentFrame],
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImages[imageIndex],
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_NEAREST);

  // === 4. outImage: TRANSFER_SRC_OPTIMAL -> GENERAL (for next compute pass)
  // ===
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  // === 5. swapchain image: TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR ===
  swapBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  swapBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  swapBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  swapBarrier.dstAccessMask = 0;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &swapBarrier);

  vkEndCommandBuffer(commandBuffer);
}

void Renderer::recordComputeCommandbuffer(VkCommandBuffer &commandbuffer) {

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(commandbuffer, &beginInfo);

  vkCmdBindPipeline(commandbuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    computePipeline);
  VkDescriptorSet bindDesciptorSets[] = {
      descriptorSets[currentFrame],
      skybox.skyboxDescriptorSet}; // {ubo, output image} , {cubemap sampler}
  vkCmdBindDescriptorSets(commandbuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          computePipelineLayout, 0, 2, bindDesciptorSets, 0,
                          nullptr);

  float currentTime = glfwGetTime();
  vkCmdPushConstants(commandbuffer, computePipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float),
                     &currentTime);

  // 1 work load = 16 x 16 x 1 invocations
  uint32_t dispatchX =
      static_cast<uint32_t>(ceil(swapchainImageExtent.width / 16.0f));
  uint32_t dispatchY =
      static_cast<uint32_t>(ceil(swapchainImageExtent.height / 16.0f));

  vkCmdDispatch(commandbuffer, dispatchX, dispatchY, 1);

  vkEndCommandBuffer(commandbuffer);
}

uint32_t Renderer::findMemoryType(uint32_t typeFilter,
                                  VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
    if (typeFilter & (1 << i) &&
        (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type!");
}

void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usageFlags,
                            VkMemoryPropertyFlags properties, VkBuffer &buffer,
                            VkDeviceMemory &bufferMemory) {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usageFlags;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

  VkMemoryRequirements memReqs{};
  vkGetBufferMemoryRequirements(device, buffer, &memReqs);

  VkMemoryAllocateInfo mallocInfo{};
  mallocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mallocInfo.allocationSize = size;
  mallocInfo.memoryTypeIndex =
      findMemoryType(memReqs.memoryTypeBits, properties);
  vkAllocateMemory(device, &mallocInfo, nullptr, &bufferMemory);

  vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

VkCommandBuffer Renderer::beginSingleTimeCommands() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer cmdbuf;
  vkAllocateCommandBuffers(device, &allocInfo, &cmdbuf);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmdbuf, &beginInfo);

  return cmdbuf;
}

void Renderer::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

  vkDeviceWaitIdle(device);
  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void Renderer::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
  auto cmdbuf = beginSingleTimeCommands();

  VkBufferCopy region{};
  region.size = size;
  region.srcOffset = 0;
  region.dstOffset = 0;
  vkCmdCopyBuffer(cmdbuf, src, dst, 1, &region);

  endSingleTimeCommands(cmdbuf);
}

// void Renderer::createVertexBuffer(std::vector<Vertex> &vertices) {
//   VkDeviceSize bufferSize = sizeof(Vertex) * vertices.size();

//   VkBuffer stagingBuffer;
//   VkDeviceMemory stagingBufferMemory;
//   createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
//                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
//                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
//                stagingBuffer, stagingBufferMemory);

//   void *data;
//   vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
//   memcpy(data, vertices.data(), bufferSize);
//   vkUnmapMemory(device, stagingBufferMemory);

//   createBuffer(
//       bufferSize,
//       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
//       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

//   copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

//   vkDestroyBuffer(device, stagingBuffer, nullptr);
//   vkFreeMemory(device, stagingBufferMemory, nullptr);
// }

// void Renderer::createIndexBuffer(std::vector<uint16_t> &indices) {
//   VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

//   VkBuffer stagingBuffer;
//   VkDeviceMemory stagingBufferMemory;
//   createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
//                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
//                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
//                stagingBuffer, stagingBufferMemory);

//   void *data;
//   vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
//   memcpy(data, indices.data(), bufferSize);
//   vkUnmapMemory(device, stagingBufferMemory);

//   createBuffer(
//       bufferSize,
//       VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
//       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

//   copyBuffer(stagingBuffer, indexBuffer, bufferSize);

//   vkDestroyBuffer(device, stagingBuffer, nullptr);
//   vkFreeMemory(device, stagingBufferMemory, nullptr);
// }

template <typename T>
void Renderer::createUniformBuffers(size_t count,
                                    std::vector<VkBuffer> &buffers,
                                    std::vector<VkDeviceMemory> &buffersMemory,
                                    std::vector<void *> &pData) {
  VkDeviceSize bufferSize = sizeof(T);

  buffers.resize(count);
  buffersMemory.resize(count);
  pData.resize(count);

  for (size_t i = 0; i < count; i++) {
    createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 buffers[i], buffersMemory[i]);

    vkMapMemory(device, buffersMemory[i], 0, bufferSize, 0, &pData[i]);
  }
}

// void Renderer::clearUBO() {
//   ubo.view = glm::lookAt(cam.cameraPos, glm::vec3(0.0, 0.0, -1.0),
//                          glm::vec3(0.0, 1.0, 0.0));
//   ubo.proj = glm::perspective(glm::radians(45.0f),
//                               swapchainImageExtent.width /
//                                   (float)swapchainImageExtent.height,
//                               0.1f, 10.0f);
//   ubo.proj[1][1] *= -1;
// }

void Renderer::clearCameraUBO() {
  camUbo.viewInv = glm::inverse(glm::lookAt(
      cam.cameraPos, glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, 1.0, 0.0)));

  glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                    swapchainImageExtent.width /
                                        (float)swapchainImageExtent.height,
                                    0.1f, 10.0f);
  proj[1][1] *= -1;
  camUbo.projInv = glm::inverse(proj);
}

void Renderer::updateUniformBuffer(uint32_t currentFrame, bool resized) {

  // ubo.view = cam.getViewMatrix();
  camUbo.viewInv = cam.getViewMatrixInv();

  if (resized) {
    glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                      swapchainImageExtent.width /
                                          (float)swapchainImageExtent.height,
                                      0.1f, 1000.0f);
    proj[1][1] *= -1;
    camUbo.projInv = glm::inverse(proj);
  }

  // memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
  memcpy(cameraBuffersMapped[currentFrame], &camUbo, sizeof(camUbo));
}

// void Renderer::createShaderStorageBuffers() {
//     shaderStorageBuffers.resize(MAX_FRAME_IN_FLIGHT);
//     shaderStorageBuffersMemory.resize(MAX_FRAME_IN_FLIGHT);

//     std::default_random_engine rndEngine((unsigned)time(nullptr));
//     std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

//     std::vector<Particle> particles(PARTICLE_COUNT);
//     for (Particle &particle : particles) {
//         float r = 0.25f * sqrt(rndDist(rndEngine));
//         float theta = rndDist(rndEngine) * 2 * 3.14159265358979323846; // 0 ~
//         2pi float x = r * cos(theta) * DEFAULT_HEIGHT / DEFAULT_WIDTH; float
//         y = r * sin(theta); particle.position = glm::vec2(x, y);
//         particle.velocity = glm::normalize(glm::vec2(x, y)) * 0.00025f;
//         particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine),
//         rndDist(rndEngine), 1.0f);
//     }

//     VkDeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;
//     VkBuffer stagingBuffer;
//     VkDeviceMemory stagingBufferMemory;
//     createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
//                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
//                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
//                 stagingBufferMemory);

//     void *data;
//     vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
//     memcpy(data, particles.data(), (size_t)bufferSize);
//     vkUnmapMemory(device, stagingBufferMemory);

//     for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
//         createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
//         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
//                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
//                     shaderStorageBuffers[i], shaderStorageBuffersMemory[i]);

//         copyBuffer(stagingBuffer, shaderStorageBuffers[i], bufferSize);
//     }

//     vkDestroyBuffer(device, stagingBuffer, nullptr);
//     vkFreeMemory(device, stagingBufferMemory, nullptr);
// }

void Renderer::createDescriptorPool() {
  std::array<VkDescriptorPoolSize, 3> poolSizes;
  // UBO (set = 0), per frame -> 2
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[0].descriptorCount = MAX_FRAME_IN_FLIGHT;

  // Sampler for skybox (set = 1), 1
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[1].descriptorCount = 1;

  // output image (set = 0), per frame -> 2
  poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  poolSizes[2].descriptorCount = MAX_FRAME_IN_FLIGHT;

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = poolSizes.size();
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets =
      MAX_FRAME_IN_FLIGHT + 1; // set 0 -> 2개, set 1 -> 1개, 총 3개
  vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
}

void Renderer::createDescriptorSetLayout() {

  std::array<VkDescriptorSetLayoutBinding, 2> bindings;
  bindings[0].binding = 0;
  bindings[0].descriptorCount = 1;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings[0].pImmutableSamplers = nullptr;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[1].binding = 1;
  bindings[1].descriptorCount = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[1].pImmutableSamplers = nullptr;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  info.bindingCount = bindings.size();
  info.pBindings = bindings.data();

  vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptorSetLayout);
}

void Renderer::createDescriptorSets() {
  std::vector<VkDescriptorSetLayout> layouts(MAX_FRAME_IN_FLIGHT,
                                             descriptorSetLayout);

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool;
  allocInfo.descriptorSetCount = layouts.size();
  allocInfo.pSetLayouts = layouts.data();

  descriptorSets.resize(MAX_FRAME_IN_FLIGHT);
  chk(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()),
      "vkAllocateDescriptorSets");

  for (size_t i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {

    VkDescriptorBufferInfo camInfo{};
    camInfo.buffer = cameraBuffers[i];
    camInfo.offset = 0;
    camInfo.range = sizeof(CameraUBO);

    VkDescriptorImageInfo outImageInfo{};
    outImageInfo.sampler = VK_NULL_HANDLE;
    outImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outImageInfo.imageView = outImageViews[i];

    std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].dstSet = descriptorSets[i];
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].pBufferInfo = &camInfo;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].dstSet = descriptorSets[i];
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrites[1].pImageInfo = &outImageInfo;

    // vkUpdateDescriptorSets를 호출하는 순간
    // 해당 dstSet + dstBinding + dstArrayElement 위치에
    // VkDescriptorImageInfo 내용이 복사된다.
    // shader에서의 set은 binding한 descriptor set 순서와 동일
    vkUpdateDescriptorSets(device, 2, descriptorWrites.data(), 0, nullptr);
  }
}

void Renderer::cleanupSwapchain() {
  for (VkFramebuffer fb : framebuffers) {
    vkDestroyFramebuffer(device, fb, nullptr);
  }

  for (VkImageView view : swapchainImageViews) {
    vkDestroyImageView(device, view, nullptr);
  }

  // Destroy output images (they are sized to swapchain extent)
  for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
    vkDestroyImageView(device, outImageViews[i], nullptr);
    vkDestroyImage(device, outImages[i], nullptr);
    vkFreeMemory(device, outImagesMemory[i], nullptr);
  }

  vkDestroySwapchainKHR(device, swapchain, nullptr);
  vkDestroyImageView(device, depthImageView, nullptr);
  vkDestroyImage(device, depthImage, nullptr);
  vkFreeMemory(device, depthImageMemory, nullptr);
}

void Renderer::recreateSwapchain(uint32_t imageIndex) {
  int width = 0, height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  while (width == 0 || height == 0) {
    if (glfwWindowShouldClose(window))
      return;

    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }

  vkDeviceWaitIdle(device);

  cleanupSwapchain();

  if (imageIndex != -1) {
    vkDestroySemaphore(this->device, renderFinishedSemaphores[imageIndex],
                       nullptr);
    VkSemaphoreCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0};
    vkCreateSemaphore(this->device, &createInfo, nullptr,
                      &renderFinishedSemaphores[imageIndex]);
  }

  createSwapchain(); // <- 여기서 extent 업데이트
  createDepthResources();
  createImageViews();
  createFramebuffers();
  createOutputImages();

  // Update existing descriptor sets with new outImageViews
  // (cannot re-allocate since pool lacks FREE_DESCRIPTOR_SET_BIT)
  for (size_t i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
    VkDescriptorImageInfo outImageInfo{};
    outImageInfo.sampler = VK_NULL_HANDLE;
    outImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    outImageInfo.imageView = outImageViews[i];

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.dstSet = descriptorSets[i];
    descriptorWrite.dstBinding = 1;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrite.pImageInfo = &outImageInfo;

    vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
  }

  for (size_t i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
    updateUniformBuffer(i, true);
  }
}

void Renderer::transitionImageLayout(
    VkImage image, VkFormat format, VkImageLayout oldLayout,
    VkImageLayout newLayout, uint32_t mipLevels, uint32_t layerCount,
    VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask,
    VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {

  VkCommandBuffer cmdBuf = beginSingleTimeCommands();

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.image = image;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                              .baseMipLevel = 0,
                              .levelCount = mipLevels,
                              .baseArrayLayer = 0,
                              .layerCount = layerCount};
  barrier.srcAccessMask = srcAccessMask;
  barrier.dstAccessMask = dstAccessMask;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

  vkCmdPipelineBarrier(cmdBuf, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1,
                       &barrier);

  endSingleTimeCommands(cmdBuf);
}

void Renderer::createImage(uint32_t width, uint32_t height,
                           VkFormat imageFormat, VkImageCreateFlags flags,
                           uint32_t mipLevels, uint32_t arrayLayers,
                           VkImageTiling tiling, VkImageUsageFlags imageUsage,
                           VkMemoryPropertyFlags properties, VkImage &image,
                           VkDeviceMemory &imageMemory) {

  VkImageCreateInfo imgCreateInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = flags,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = imageFormat,
      .extent = {.width = width, .height = height, .depth = 1},
      .mipLevels = mipLevels,
      .arrayLayers = arrayLayers,
      .samples = VK_SAMPLE_COUNT_1_BIT, // This is only relevant for images that
                                        // will be used as attachments, so stick
                                        // to one sample
      .tiling = tiling,
      .usage = imageUsage,
      .sharingMode =
          VK_SHARING_MODE_EXCLUSIVE, // The image will only be used by one queue
                                     // family: the one that supports graphics
                                     // (and therefore also) transfer
                                     // operations.
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

  vkCreateImage(device, &imgCreateInfo, nullptr, &image);

  VkMemoryRequirements memReqs;
  vkGetImageMemoryRequirements(device, image, &memReqs);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memReqs.memoryTypeBits, properties);
  vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory);

  vkBindImageMemory(device, image, imageMemory, 0);
}

VkImageView Renderer::createImageView(VkImage image, VkFormat format,
                                      VkImageViewType viewType,
                                      VkImageAspectFlags aspectFlags,
                                      uint32_t mipLevels, uint32_t layerCount) {

  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image;
  viewInfo.viewType = viewType;
  viewInfo.format = format;
  viewInfo.subresourceRange = {.aspectMask = aspectFlags,
                               .baseMipLevel = 0,
                               .levelCount = mipLevels,
                               .baseArrayLayer = 0,
                               .layerCount = layerCount};

  VkImageView imageView;
  chk(vkCreateImageView(device, &viewInfo, nullptr, &imageView),
      "vkCreateImageView");

  return imageView;
}

VkFormat Renderer::findSupportedFormat(const std::vector<VkFormat> &candidates,
                                       VkImageTiling tiling,
                                       VkFormatFeatureFlags features) {

  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physDev, format, &props);

    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  throw std::runtime_error("Failed to find supported format!");
}

VkFormat Renderer::findDepthFormat() {
  return findSupportedFormat(
      {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
       VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void Renderer::createDepthResources() {
  VkFormat depthFormat = findDepthFormat();
  createImage(swapchainImageExtent.width, swapchainImageExtent.height,
              depthFormat, 0, 1, 1, VK_IMAGE_TILING_OPTIMAL,
              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage,
              depthImageMemory);
  depthImageView =
      createImageView(depthImage, depthFormat, VK_IMAGE_VIEW_TYPE_2D,
                      VK_IMAGE_ASPECT_DEPTH_BIT, 1, 1);
}

void Renderer::createSkybox() { skybox.init(this, CUBEMAP_TEXTURE_PATH); }

void Renderer::processInput1(GLFWwindow *window, Camera &cam, float dt) {

  cam.update(dt, glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS,
             glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS,
             glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS,
             glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS,
             glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS,
             glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS,
             glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS,
             glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS,
             glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS,
             glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS);
}

void Renderer::createOutputImages() {
  outImages.resize(MAX_FRAME_IN_FLIGHT);
  outImagesMemory.resize(MAX_FRAME_IN_FLIGHT);
  outImageViews.resize(MAX_FRAME_IN_FLIGHT);
  for (int i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
    createImage(swapchainImageExtent.width, swapchainImageExtent.height,
                VK_FORMAT_R8G8B8A8_UNORM, 0, 1, 1, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outImages[i],
                outImagesMemory[i]);

    transitionImageLayout(outImages[i], VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 1,
                          1, VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    outImageViews[i] =
        createImageView(outImages[i], VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1);
  }
}

// _______________________________________________________________
static void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
  auto app = reinterpret_cast<Renderer *>(glfwGetWindowUserPointer(window));
  app->framebufferResized = true;
}
