#include "include/skybox.h"
#include "include/renderer.h"
#include "include/utils.h"
#include <cstring>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

const VkFormat textureFormat = VK_FORMAT_R8G8B8A8_SRGB;

void Skybox::destroy() {

  //   vkDestroyPipeline(renderer->device, this->skyboxGraphicsPipeline,
  //   nullptr); vkDestroyPipelineLayout(renderer->device,
  //   skyboxGraphicsPipelineLayout,
  //                           nullptr);
  vkDestroyDescriptorSetLayout(renderer->device,
                               this->skyboxDescriptorSetLayout, nullptr);
  vkDestroySampler(renderer->device, this->skyboxTextureSampler, nullptr);
  vkDestroyImageView(renderer->device, this->skyboxTextureView, nullptr);
  vkDestroyImage(renderer->device, this->skyboxTexture, nullptr);
  vkFreeMemory(renderer->device, this->skyboxTextureMemory, nullptr);
}

void Skybox::init(Renderer *renderer, const char *texture_dir) {

  this->renderer = renderer;

  const std::vector<std::string> faceFiles = {"_px.png", "_nx.png", "_py.png",
                                              "_ny.png", "_pz.png", "_nz.png"};

  stbi_uc *pixels[6];
  int texWidth, texHeight, texChannels;
  VkDeviceSize layerSize = 0;

  for (int i = 0; i < 6; i++) {
    std::string path = std::string(texture_dir) + faceFiles[i];
    pixels[i] = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels,
                          STBI_rgb_alpha);
    if (!pixels[i]) {
      throw std::runtime_error("Failed to load texture image: " + path);
    }
    if (i == 0) {
      layerSize = texWidth * texHeight * 4;
    }
  }

  VkDeviceSize totalImageSize = layerSize * 6;

  // create image
  renderer->createImage(
      texWidth, texHeight, textureFormat, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
      1, 6, VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
          VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, this->skyboxTexture,
      this->skyboxTextureMemory);

  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = this->skyboxTexture;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  viewInfo.format = textureFormat;
  viewInfo.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .baseMipLevel = 0,
                               .levelCount = 1,
                               .baseArrayLayer = 0,
                               .layerCount = 6};
  vkCreateImageView(renderer->device, &viewInfo, nullptr,
                    &this->skyboxTextureView);

  // create staging buffer
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  renderer->createBuffer(totalImageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stagingBuffer, stagingBufferMemory);

  // copy pixels to staging buffer
  void *data;
  vkMapMemory(renderer->device, stagingBufferMemory, 0, totalImageSize, 0,
              &data);
  for (int i = 0; i < 6; i++) {
    memcpy(static_cast<uint8_t *>(data) + (layerSize * i), pixels[i],
           layerSize);
    stbi_image_free(pixels[i]);
  }
  vkUnmapMemory(renderer->device, stagingBufferMemory);

  std::vector<VkBufferImageCopy> regions;
  for (int i = 0; i < 6; i++) {
    VkBufferImageCopy region{};
    region.bufferOffset = layerSize * i;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .mipLevel = 0,
                               .baseArrayLayer = static_cast<uint32_t>(i),
                               .layerCount = 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {.width = static_cast<uint32_t>(texWidth),
                          .height = static_cast<uint32_t>(texHeight),
                          .depth = 1};
    regions.push_back(region);
  }

  renderer->transitionImageLayout(
      this->skyboxTexture, textureFormat, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 6, VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkCommandBuffer cmdbuf = renderer->beginSingleTimeCommands();
  vkCmdCopyBufferToImage(cmdbuf, stagingBuffer, this->skyboxTexture,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         static_cast<uint32_t>(regions.size()), regions.data());
  renderer->endSingleTimeCommands(cmdbuf);

  renderer->transitionImageLayout(
      this->skyboxTexture, textureFormat, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 6,
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

  vkDestroyBuffer(renderer->device, stagingBuffer, nullptr);
  vkFreeMemory(renderer->device, stagingBufferMemory, nullptr);

  createSampler();
  createDescriptorSets();
  // createGraphicsPipeline();
}

void Skybox::createSampler() {
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;

  VkPhysicalDeviceFeatures supportedFeatures{};
  vkGetPhysicalDeviceFeatures(renderer->physDev, &supportedFeatures);
  if (supportedFeatures.samplerAnisotropy) {
    VkPhysicalDeviceProperties devProps{};
    vkGetPhysicalDeviceProperties(renderer->physDev, &devProps);

    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = devProps.limits.maxSamplerAnisotropy;
  }
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.minLod = 0.0f;
  samplerInfo.maxLod = 0.0f;
  vkCreateSampler(renderer->device, &samplerInfo, nullptr,
                  &this->skyboxTextureSampler);
}

void Skybox::createDescriptorSets() {

  VkDescriptorSetLayoutBinding samplerBinding{};
  samplerBinding.binding = 0;
  samplerBinding.descriptorCount = 1;
  samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &samplerBinding;
  vkCreateDescriptorSetLayout(renderer->device, &layoutInfo, nullptr,
                              &this->skyboxDescriptorSetLayout);

  VkDescriptorSetAllocateInfo descSetAllocInfo{};
  descSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descSetAllocInfo.descriptorPool = renderer->descriptorPool;
  descSetAllocInfo.descriptorSetCount = 1;
  descSetAllocInfo.pSetLayouts = &skyboxDescriptorSetLayout;
  vkAllocateDescriptorSets(renderer->device, &descSetAllocInfo,
                           &this->skyboxDescriptorSet);

  VkDescriptorImageInfo imageInfo{};
  imageInfo.sampler = this->skyboxTextureSampler;
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imageInfo.imageView = this->skyboxTextureView;

  VkWriteDescriptorSet descriptorWrite{};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  descriptorWrite.dstSet = this->skyboxDescriptorSet;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.pImageInfo = &imageInfo;

  vkUpdateDescriptorSets(renderer->device, 1, &descriptorWrite, 0, nullptr);
}

void Skybox::createGraphicsPipeline() {

  VkShaderModule vertexShaderModule =
      createShader(renderer->device, "shaders/skybox/spv/vert.spv");
  VkShaderModule fragShaderModule =
      createShader(renderer->device, "shaders/skybox/spv/frag.spv");

  // shader
  VkPipelineShaderStageCreateInfo vertexShader{};
  vertexShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertexShader.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertexShader.module = vertexShaderModule;
  vertexShader.pName = "main";

  VkPipelineShaderStageCreateInfo fragShader{};
  fragShader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShader.module = fragShaderModule;
  fragShader.pName = "main";

  VkPipelineShaderStageCreateInfo shaderStages[] = {vertexShader, fragShader};

  // vertex state
  // empty struct since vertex is in shader code (VBO isn't needed)
  VkPipelineVertexInputStateCreateInfo vertexStateInfo{};
  vertexStateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  // input assembly
  VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
  assemblyInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  assemblyInfo.primitiveRestartEnable = VK_FALSE;

  // dynamic states
  std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR};

  // viewport
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates = dynamicStates.data();

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  // Rasterizer
  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;

  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

  rasterizer.depthBiasEnable = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.0f;
  rasterizer.depthBiasClamp = 0.0f;
  rasterizer.depthBiasSlopeFactor = 0.0f;

  // Multisampling
  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading = 1.0f;
  multisampling.pSampleMask = nullptr;
  multisampling.alphaToCoverageEnable = VK_FALSE;
  multisampling.alphaToOneEnable = VK_FALSE;

  // Color blending
  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;
  colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
  colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo colorblending{};
  colorblending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorblending.logicOpEnable = VK_FALSE;
  colorblending.logicOp = VK_LOGIC_OP_COPY;
  colorblending.attachmentCount = 1;
  colorblending.pAttachments = &colorBlendAttachment;
  colorblending.blendConstants[0] = 0.0f;
  colorblending.blendConstants[1] = 0.0f;
  colorblending.blendConstants[2] = 0.0f;
  colorblending.blendConstants[3] = 0.0f;

  // depth/stencil testing
  VkPipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_FALSE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.minDepthBounds = 0.0f;
  depthStencil.maxDepthBounds = 1.0f;
  depthStencil.stencilTestEnable = VK_FALSE;
  depthStencil.front = {};
  depthStencil.back = {};

  //   VkDescriptorSetLayout layouts[] = {renderer->descriptorSetLayout,
  //                                      this->skyboxDescriptorSetLayout};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 0;
  pipelineLayoutInfo.pSetLayouts = nullptr;
  pipelineLayoutInfo.pushConstantRangeCount = 0;
  pipelineLayoutInfo.pPushConstantRanges = nullptr;
  vkCreatePipelineLayout(renderer->device, &pipelineLayoutInfo, nullptr,
                         &this->skyboxGraphicsPipelineLayout);

  VkGraphicsPipelineCreateInfo pipeInfo{};
  pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeInfo.stageCount = 2;
  pipeInfo.pStages = shaderStages;
  pipeInfo.pVertexInputState = &vertexStateInfo;
  pipeInfo.pInputAssemblyState = &assemblyInfo;
  pipeInfo.pViewportState = &viewportState;
  pipeInfo.pRasterizationState = &rasterizer;
  pipeInfo.pMultisampleState = &multisampling;
  pipeInfo.pDepthStencilState = &depthStencil;
  pipeInfo.pColorBlendState = &colorblending;
  pipeInfo.pDynamicState = &dynamicState;
  pipeInfo.layout = this->skyboxGraphicsPipelineLayout;
  pipeInfo.renderPass = renderer->renderpass;
  pipeInfo.subpass = 0;
  pipeInfo.basePipelineHandle = VK_NULL_HANDLE;
  pipeInfo.basePipelineIndex = -1;

  vkCreateGraphicsPipelines(renderer->device, nullptr, 1, &pipeInfo, nullptr,
                            &this->skyboxGraphicsPipeline);

  vkDestroyShaderModule(renderer->device, vertexShaderModule, nullptr);
  vkDestroyShaderModule(renderer->device, fragShaderModule, nullptr);
}