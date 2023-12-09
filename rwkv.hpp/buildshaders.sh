# bash shell script to compile all shaders in ./shaders into ./build
# iterate through ./shaders and compile all shaders into ./build    
#

# create ./build/ directory if not exist
mkdir -p ./build/

# create ./build/shaders/ directory if not exist  
mkdir -p ./build/shaders/  

# iterate through ./shaders and compile all shaders into ./build
for shader in ./shaders/*.glsl
do
    glslc $shader -o ./build/shaders/${shader##*/}.spv
done