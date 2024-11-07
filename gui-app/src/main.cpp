#include <tiny_websockets/client.hpp>
#include <iostream>

using namespace websockets;

int main() {
  // connect to host
  WebsocketsClient client;
  printf("Connecting to host\n");
  client.connect("http://localhost:10000/");
  
  // handle messages
  client.onMessage([](WebsocketsMessage message) {
    std::cout << "Got: " << message.data() << std::endl;
  });
  
  // handle events
  client.onEvent([](WebsocketsEvent event, std::string data) {
    printf("Event: %d Data: %s\n", event, data.c_str());
    // Handle "Pings", "Pongs" and other events 
  });
  
  printf("Sending message\n");
  // send a message
  client.send("Hi Server!");
  
  while(client.available()) {
    // wait for server messages and events
    client.poll();
  }
}