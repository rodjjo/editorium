/*
 * Copyright (C) 2023 by Rodrigo Antonio de Araujo
 */
#ifndef SRC_WINDOWS_MAIN_MENU_H_
#define SRC_WINDOWS_MAIN_MENU_H_

#include <functional>
#include <string>
#include <memory>
#include <list>

#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Menu_Item.H>
#include <FL/Fl_Multi_Label.H>

#include "components/xpm/xpm.h"
#include "messagebus/messagebus.h"

namespace editorium
{

typedef std::function<void()> callback_t;


struct MenuItem {
   std::string path;
   std::shared_ptr<Fl_RGB_Image> icon;
   std::string text;
   Fl_Multi_Label label;
   event_id_t id;
};

typedef MenuItem menu_item_t;
   
class MainMenu: public Fl_Menu_Bar {
 public:
    MainMenu(int w, int h);
    virtual ~MainMenu();
    int handle(int value) override;
    void addItem(event_id_t id, const char *path, const char* label, const char* shortcut=NULL, int flags=0, xpm::xpm_t icon=xpm::no_image);
    void enablePath(const char *path, bool enabled);
    
 protected:
    static void menuItemCb(Fl_Widget *widget, void *user_data);

 private:
    Fl_Menu_Item *component(const std::string& path);

 private:
    std::list<std::shared_ptr<menu_item_t> > items_;
};
    
}  // namespace dexpert


#endif  // SRC_WINDOWS_MAIN_MENU_H_
