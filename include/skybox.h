#pragma once

#include <vulkan/vulkan.h>
// #include "include/renderer.h"

class Renderer;

class Skybox {

public:

    void init(Renderer *renderer, const char* texture_path);
    void destroy();

    VkPipelineLayout skyboxGraphicsPipelineLayout;
    VkPipeline skyboxGraphicsPipeline;
    VkDescriptorSet skyboxDescriptorSet;
    VkDescriptorSetLayout skyboxDescriptorSetLayout;

private:
    void createSampler();
    void createDescriptorSets();
    void createGraphicsPipeline();

    Renderer *renderer;
    VkImage skyboxTexture;    
    VkDeviceMemory skyboxTextureMemory;
    VkImageView skyboxTextureView;
    VkSampler skyboxTextureSampler;
};