import sys
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject, GLib
import sys
from matplotlib.backends.backend_gtk3 import (
    NavigationToolbar2GTK3 as NavigationToolbar)
from matplotlib.backends.backend_gtk3agg import (
    FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib.figure import Figure
import numpy as np
from PRF_Controller import PRF_Controller
sysArg = sys.argv

class NavigationToolbar(NavigationToolbar):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Save')]


class Cursor(object):
    def __init__(self):
        self.ax = ''
        self.lx = ''
        self.ly = ''
        self.tstObj = None
        self.coodsList = None

    def set_ax(self, ax, tstObj, ctrlObj):
        self.ax = ax
        self.lx = ax.axhline(color='b')  # the horiz line
        self.ly = ax.axvline(color='r')  # the vert line
        self.tstObj = tstObj
        self.coodsList = ctrlObj
        self.pltX=1
        self.pltY=2
        if len(self.tstObj) ==5:
            self.pltX=2
            self.pltY=3


    def mouse_Click(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)
        self.tstObj[self.pltX].clear()
        self.tstObj[self.pltY].clear()
        profileList = self.coodsList.getProfiles(x, y)
        self.tstObj[self.pltX].plot(profileList[0],profileList[1],color='b')
        self.tstObj[self.pltX].set_title("X Profile", fontsize=24)
        self.tstObj[self.pltX].set_xlabel("X", fontsize=24)
        self.tstObj[self.pltX].set_ylabel("Z", fontsize=24)
                
        self.tstObj[self.pltY].plot(profileList[2],profileList[3],color='r')
        self.tstObj[self.pltY].set_title("Y Profile", fontsize=24)
        self.tstObj[self.pltY].set_xlabel("Y", fontsize=24)
        self.tstObj[self.pltY].set_ylabel("Z", fontsize=24)

        
        self.ax.figure.canvas.draw()
        self.tstObj[self.pltX].figure.canvas.draw()
        self.tstObj[self.pltY].figure.canvas.draw()
        
        

    

class PRF_GUI(Gtk.Window):
    def __init__(self):
        self.FPath = ''
        self.FName = ''
        self.selectedAnalysis = ''
        self.selectedFilt = ''
        self.analysisList = ['Plot Only','Roughness']
        self.analysis_store = Gtk.ListStore(str)
        self.filtList = ['None','Gaussian Low Pass', 'Gaussian High Pass']
        self.filt_store = Gtk.ListStore(str)
        self.ctrlObj = None

        #Set the main window properties
        Gtk.Window.__init__(self, title="PRF ASCII Analysis")
        self.set_resizable(False)
        self.set_border_width(10)
        self.settings = Gtk.Settings.get_default()

        #Create a grid to store all the objects
        self.grid = Gtk.Grid(column_homogeneous=False, column_spacing=10, row_spacing=10)
        self.add(self.grid)
        

        #Create the browse button and set properties
        self.openAsciiLabel = Gtk.Label(label="Open ASCII: ")
        self.grid.attach(self.openAsciiLabel, 0, 0, 1, 1)
        self.openEntry = Gtk.Entry()
        self.openEntry.set_editable(False)
        self.grid.attach(self.openEntry, 1, 0, 80, 1)
        self.browseBtn = Gtk.Button(label="Browse")
        self.browseBtn.set_size_request(50, 20)
        self.browseBtn.connect("clicked", self.on_browse_button_clicked)
        self.grid.attach(self.browseBtn, 81, 0, 1, 1)


        #Create Tip/Tilt Removal toggle button
        self.tipTiltBtn = Gtk.CheckButton(label="Remove Tilt")
        self.tipTiltBtn.connect("toggled", self.on_tiptilt_button_clicked)
        self.grid.attach(self.tipTiltBtn, 31, 1, 1, 1)


        #Create Theme toggle button
        self.themeBtn = Gtk.CheckButton(label="Dark-Theme")
        self.themeBtn.connect("toggled", self.on_themeBtn_button_clicked)
        self.grid.attach(self.themeBtn, 82, 0, 1, 1)


        #Create the analysis options and set properties
        self.analysisLabel = Gtk.Label(label="Analysis: ")
        self.grid.attach(self.analysisLabel, 0, 1, 1, 1)
        self.analysisCombobox = Gtk.ComboBox.new_with_model(self.analysis_store)
        self.analysisCombobox.connect("changed", self.on_analysisCombobox_changed)
        #self.filterCombobox.set_sensitive(False)
        renderer_text = Gtk.CellRendererText()
        self.analysisCombobox.pack_start(renderer_text, True)
        self.analysisCombobox.add_attribute(renderer_text, "text", 0)
        self.grid.attach(self.analysisCombobox, 1, 1, 30, 1)
        for item in self.analysisList: self.analysis_store.append([item])
        self.analysisCombobox.set_active(0)


        #Create the filter options and set properties
        self.filterLabel = Gtk.Label(label="Filter: ")
        self.grid.attach(self.filterLabel, 0, 2, 1, 1)
        self.filterCombobox = Gtk.ComboBox.new_with_model(self.filt_store)
        self.filterCombobox.connect("changed", self.on_filtCombobox_changed)
        #self.filterCombobox.set_sensitive(False)
        renderer_text = Gtk.CellRendererText()
        self.filterCombobox.pack_start(renderer_text, True)
        self.filterCombobox.add_attribute(renderer_text, "text", 0)
        self.grid.attach(self.filterCombobox, 1, 2, 30, 1)
        for item in self.filtList: self.filt_store.append([item])
        self.filterCombobox.set_active(0)
        

        #Create the close button and set properties
        self.closeBtn = Gtk.Button(label="Close")        
        self.closeBtn.set_size_request(50, 20)
        self.closeBtn.connect("clicked", self.on_close_button_clicked)
        self.grid.attach(self.closeBtn, 99, 80, 1, 1)


        #Create the clear button and set properties
        self.clearBtn = Gtk.Button(label="Clear")        
        self.clearBtn.set_size_request(50, 20)
        self.clearBtn.connect("clicked", self.on_clear_button_clicked)
        self.grid.attach(self.clearBtn, 98, 80, 1, 1)


        #Create the save all button and set properties
        self.saveBtn = Gtk.Button(label="Save All")        
        self.saveBtn.set_size_request(50, 20)
        self.saveBtn.connect("clicked", self.on_save_button_clicked)
        self.grid.attach(self.saveBtn, 97, 80, 1, 1)



        #Create an empty matplotlib figure
        self.vbox = Gtk.VBox()
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.vbox.pack_start(self.canvas, True, True, 0)
        self.grid.attach(self.vbox, 0, 3, 100, 70) #(Obj, col, row, len, width)
        # Create toolbar
        toolbar = NavigationToolbar(self.canvas, self)
        self.vbox.pack_start(toolbar, False, False, 10)
        self.cursor = Cursor()

        #Gets system args passed to .exe (for example: someone changes the .asc to open with this script, it will use the arg to open the .asc file).
        #If .asc file gets double clicked, the file will automatically get loaded and plots created.
        if len(sysArg) >= 2:
            self.sysArgOpen()


    def on_browse_button_clicked(self, widget):
        dialog = Gtk.FileChooserDialog(title="Open..",
                                       parent=None,
                                       action=Gtk.FileChooserAction.OPEN)
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )

        #Add a filter to only find ASCII files
        filter = Gtk.FileFilter()
        filter.set_name('.asc')
        filter.add_pattern('*.asc')
        dialog.add_filter(filter)
        response = dialog.run()

        
        if response == Gtk.ResponseType.OK:
            if len(dialog.get_filename().split('.asc')) == 2:
                try:
                    self.FPath = '\\'.join(dialog.get_filename().split('\\')[:-1])+'\\'
                    self.FName = dialog.get_filename().split('\\')[-1]
                    self.openEntry.set_text(self.FName)

                    #Open the file and create a new figure
                    self.ctrlObj = PRF_Controller(self.FName, self.FPath, self.selectedAnalysis, self.tipTiltBtn.get_active(), self.selectedFilt)
                    self.fig = self.ctrlObj.getFigObj()
                    self.ReplaceFigure(self.fig)
                except:
                    message  = self.userMessageDialog('File Error', 'The file selected uses the wrong naming convention.', Gtk.MessageType.ERROR)
                    message.run()
                    message.destroy()
                    self.openEntry.set_text("")
                    dialog.destroy()
                
        dialog.destroy()

    def on_tiptilt_button_clicked(self, widget):
        if self.openEntry.get_text() != '':
                self.ctrlObj.updateProperties(self.tipTiltBtn.get_active(), self.selectedFilt)
                self.fig = self.ctrlObj.getFigObj()
                self.ReplaceFigure(self.fig)

    def on_themeBtn_button_clicked(self, widget):
        self.settings.set_property("gtk-application-prefer-dark-theme", self.themeBtn.get_active())


    def on_close_button_clicked(self, widget):
        Gtk.Window.destroy(self)
        Gtk.main_quit()


    def on_clear_button_clicked(self, widget):
        self.openEntry.set_text("")
        self.analysisCombobox.set_active(0)
        self.filterCombobox.set_active(0)
        self.tipTiltBtn.set_active(False)
        for element in self.vbox.get_children():
            self.vbox.remove(element)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)  # a Gtk.DrawingArea        
        self.vbox.pack_start(self.canvas, True, True, 0)
        toolbar = NavigationToolbar(self.canvas, self)
        self.vbox.pack_start(toolbar, False, False, 10)
        self.show_all()
      


    def on_save_button_clicked(self, widget):
        self.ctrlObj.getFigObj(saveFig=True)
        message  = self.userMessageDialog('Save Files', 'Files Successfully Saved!!!',
                                          Gtk.MessageType.INFO)
        message.run()
        message.destroy()


    def on_analysisCombobox_changed(self, combo):
        tree_iter = combo.get_active_iter()
        if tree_iter is not None:
            model = combo.get_model()
            self.selectedAnalysis = model[tree_iter][0]
            if self.openEntry.get_text() != '':
                self.ctrlObj = PRF_Controller(self.FName, self.FPath, self.selectedAnalysis, self.tipTiltBtn.get_active(), self.selectedFilt)
                self.fig = self.ctrlObj.getFigObj()
                self.ReplaceFigure(self.fig)
                

    def on_filtCombobox_changed(self, combo):
        tree_iter = combo.get_active_iter()
        if tree_iter is not None:
            model = combo.get_model()
            self.selectedFilt = model[tree_iter][0]
            if self.openEntry.get_text() != '':
                self.ctrlObj.updateProperties(self.tipTiltBtn.get_active(), self.selectedFilt)
                self.fig = self.ctrlObj.getFigObj()
                self.ReplaceFigure(self.fig)
            
    #Method used to remove all the children from the vbox so we can fill it with new objects
    def ReplaceFigure(self, fig):
        for element in self.vbox.get_children():
            self.vbox.remove(element)
        self.canvas = FigureCanvas(self.fig)  # a Gtk.DrawingArea        
        self.vbox.pack_start(self.canvas, True, True, 0)
        toolbar = NavigationToolbar(self.canvas, self)
        self.vbox.pack_start(toolbar, False, False, 10)
        self.cursor.set_ax(self.fig.get_axes()[0], self.fig.get_axes(), self.ctrlObj)
        self.fig.canvas.mpl_connect('button_press_event', self.cursor.mouse_Click)
        self.show_all()


                
    #Method used to open a file if an argument is passed to the .exe
    def sysArgOpen(self):
        if len(sysArg[1].split('.asc')) == 2:
                self.FPath = '\\'.join(sysArg[1].split('\\')[:-1])+'\\'
                self.FName = sysArg[1].split('\\')[-1]
                self.openEntry.set_text(self.FName)
                #Open the file and create a new figure
                self.ctrlObj = PRF_Controller(self.FName, self.FPath, self.selectedAnalysis, self.tipTiltBtn.get_active(), self.selectedFilt)
                self.fig = self.ctrlObj.getFigObj()
                self.ReplaceFigure(self.fig)


    def userMessageDialog(self, messageTitle='', messageText='', messageType=Gtk.MessageType.INFO):
            message = Gtk.MessageDialog(title=messageTitle, modal=Gtk.DialogFlags.MODAL,
                                message_type=messageType,
                                buttons=Gtk.ButtonsType.CLOSE,
                                text=messageText)
            return message

win = PRF_GUI()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
