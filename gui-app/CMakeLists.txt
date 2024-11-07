cmake_minimum_required(VERSION 3.6.0 FATAL_ERROR)

project(editorium-ui C CXX)



set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FLTK_BUILD_FLUID ON)
set(FLTK_BUILD_EXAMPLES OFF)
set(FLTK_BUILD_TEST OFF)
set(OPTION_BUILD_SHARED_LIBS OFF)

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin)

add_compile_definitions(
    ASIO_STANDALONE 
    BOOST_ASIO_STANDALONE
)


add_subdirectory("dependencies/fltk")
add_subdirectory("dependencies/tiny_websockets")


include_directories(
    ${PROJECT_SOURCE_DIR}
    "${CMAKE_CURRENT_LIST_DIR}/src"
    "${CMAKE_CURRENT_LIST_DIR}/dependencies/cimg"
    "${CMAKE_CURRENT_LIST_DIR}/dependencies/nlohmann_json/include"
    "${CMAKE_CURRENT_LIST_DIR}/dependencies/tiny_websockets/tiny_websockets_lib/include"
    "${CMAKE_CURRENT_LIST_DIR}/dependencies/asio_only/asio/include"
    "${CMAKE_CURRENT_LIST_DIR}/dependencies/fltk"
    "${CMAKE_CURRENT_LIST_DIR}/build/dependencies/fltk"
)



file(GLOB Sources
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/images/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/misc/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/components/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/components/xpm/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/messagebus/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/windows/*.cpp"
)

add_executable(editorium-ui ${Sources})

target_include_directories(editorium-ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    editorium-ui 
    stdc++ 
    fltk
    fltk_forms 
    fltk_images 
    fltk_gl 
    tiny_websockets_lib
    X11
)

set_target_properties(editorium-ui
    PROPERTIES
    LINKER_LANGUAGE C
)

target_link_options(editorium-ui PRIVATE ${SUBSYSTEM_LINKER_OPTIONS})

install(TARGETS editorium-ui
        RUNTIME DESTINATION bin
        COMPONENT runtime)
