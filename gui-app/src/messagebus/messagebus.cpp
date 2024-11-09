#include <list>
#include <mutex>
#include <thread>
#include <map>
#include <FL/Fl.H>
#include "messagebus/messagebus.h"

namespace editorium
{

namespace 
{
    typedef std::list<std::pair<event_id_t, std::pair<void*, void*>> > event_queue_t;
    size_t current_subscriber_id = 0;
    std::map<event_id_t, std::list<Subscriber *> > subscribers;
    std::mutex event_mutex;
    bool event_processor_enabled = false;
    event_queue_t event_queue;
} // namespace 


void event_processor(void *) {
    event_queue_t processing;
    {
        std::unique_lock<std::mutex> lk(event_mutex);
        processing = event_queue;
        event_queue.clear();
    }
    for (auto & item : processing) {
        auto lst = subscribers.find(item.first);
        if (lst == subscribers.end()) {
            continue;
        }
        for (auto * sub: lst->second) {
            (*sub)(item.second.first, item.first, item.second.second);
        }
    }
    Fl::repeat_timeout(0.015, event_processor, NULL);
}


void publish_event(void *sender, event_id_t event, void *data) {
    std::unique_lock<std::mutex> lk(event_mutex);
    if (!event_processor_enabled) {
        Fl::add_timeout(0.001, event_processor, NULL);
    }
    event_queue.push_back({event, {sender, data}});
}

Subscriber::Subscriber(const std::list<event_id_t>& events, event_handler_t handler)  {
    handler_ = handler;
    for (auto event : events) {
        auto lst = subscribers.find(event);
        if (lst == subscribers.end()) {
            subscribers[event] = std::list<Subscriber *>({ this });
        } else {
            lst->second.push_back(this);
        }
    }
};

Subscriber::~Subscriber() {
    for (auto & item : subscribers) {
        item.second.remove(this);
    }
}


    
} // namespace editorium
