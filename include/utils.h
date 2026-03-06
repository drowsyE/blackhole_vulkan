#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <fstream>

void chk(VkResult res, const char* msg);
void populateDebugMessenger(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
VkResult createDebugUtilsMessenger(VkInstance &instance,
                                   const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                   const VkAllocationCallbacks *pAllocator,
                                   VkDebugUtilsMessengerEXT *pDebugMessenger);
void destroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks *pAllocator);
std::vector<char> readFile(const char *filename);
VkShaderModule createShader(VkDevice& device, const char* filename);