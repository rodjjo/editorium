#include "main_window.h"
#include "websocket/code.h"

int main(int argc, char **argv) {        
    editorium::ws::run_websocket();
    int result = editorium::MainWindow::dfe_run();
    editorium::ws::stop_websocket();
    return result;
}