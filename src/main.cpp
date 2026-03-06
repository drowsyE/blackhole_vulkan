#include <stdio.h>
#include <chrono>
#include <thread>
#include "include/renderer.h"


#define TARGET_FRAME_TIME 1/60 // 60 fps


int main() {
    Renderer renderer;  

    renderer.run();
    
    printf("\ntest okay\n\n");
}