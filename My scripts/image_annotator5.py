from tkinter import *
from tkinter.filedialog import askdirectory
from PIL import Image, ImageTk
import os


# TODO: make it accept img_width, img_height, etc as command line arguments

IMAGE_FOLDER = ''
OUTFILE_NAME = 'labels.csv'
TARGET_IMG_WIDTH = 224
TARGET_IMG_HEIGHT = 224

# if True: structure is dir/[subdir1, subdir2, etc] with no pics in dir itself
# if False: structure is dir/[pic1, pic2, etc]
# TODO: program this in so that the variable doesn't have to be updated
CONTAINS_SUBDIRS = True


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master  # master widget = tk window

        self.image_folder = IMAGE_FOLDER
        self.dirs = os.listdir(self.image_folder)

        if CONTAINS_SUBDIRS:
            # find one layer of subfolders
            for i, d in enumerate(self.dirs):
                if '.' not in d:
                    self.dirs[i] = os.listdir(self.image_folder + '/' + d)
                    for i2, d2 in enumerate(self.dirs[i]):
                        self.dirs[i][i2] = d + '/' + self.dirs[i][i2]

            # flatten list of lists to just a plain ol' list
            self.dirs = [item for sublist in self.dirs
                         for item in sublist
                         if type(sublist) is list]

        print(f'There are {len(self.dirs)} images to label - good luck')

        self.index = 0

        self.first_xy_added = False

        print('\nx\ty\twidth\theight\timg')
        print('------------------------')

        self.init_window()  # run init_window, which doesn't yet exist

    def init_window(self):
        self.master.title("Simple Image Annotator")  # title of master widget
        self.pack(fill=BOTH, expand=1)  # give widget full space of root window

        menu = Menu(self.master)  # creating a menu instance
        self.master.config(menu=menu)

        file = Menu(menu)  # create the file object
        file.add_command(label='Exit', command=self.client_exit)
        menu.add_cascade(label='File', menu=file)  # added "file" to our menu

        self.master.bind('<Button 1>', self.add_with_coords)  # left button
        self.master.bind('<Button 3>', self.add_without_coords)  # right button
        self.master.bind('<space>', self.skip_to_next_pic)

        self.load_new_image()

    def add_with_coords(self, event):
        # scale to be out of 100
        x = int(event.x * 100./self.img_width)
        y = int(event.y * 100./self.img_height)

        if self.first_xy_added:
            width = x - self.first_x
            height = y - self.first_y
            add_to_file(self.dirs[self.index-1], 1,
                        self.first_x, self.first_y, width, height)

            print('%i\t%i\t%i\t%i\t%s'
                  % (self.first_x, self.first_y,
                     width, height, self.dirs[self.index-1]))

            if self.index >= len(self.dirs):
                exit()

            self.load_new_image()

        else:
            self.first_x = x
            self.first_y = y
            self.first_xy_added = True

    def add_without_coords(self, event):
        add_to_file(self.dirs[self.index-1], 0, -1, -1, -1, -1)
        print('No object:\t%s' % self.dirs[self.index-1])

        if self.index >= len(self.dirs):
            exit()

        self.load_new_image()

    def skip_to_next_pic(self, event):
        if self.index >= len(self.dirs):
            exit()

        self.load_new_image()

    def client_exit(self):
        exit()

    def load_new_image(self):
        load = Image.open(self.image_folder + '/' + self.dirs[self.index])

        load = load.resize((TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT),
                           Image.ANTIALIAS)

        self.img_width = load.size[0]
        self.img_height = load.size[1]
        self.index += 1
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        img = Label(self, image=render)
        img.image = render
        img.place(x=0, y=0)

        self.first_xy_added = False


def create_initial_file(OUTFILE_NAME):
    """ ugly way of avoiding overwriting an existing file """
    if os.path.exists(OUTFILE_NAME):
        i = 0
        tmp = OUTFILE_NAME
        while os.path.exists(tmp):
            tmp = OUTFILE_NAME[0:-4] + str(i) + '.csv'
            i += 1
        OUTFILE_NAME = tmp

    with open(OUTFILE_NAME, 'w') as f:
        f.write('img,contains_obj,x,y,width,height\n')

    print('Saving file as %s' % OUTFILE_NAME)

    return OUTFILE_NAME


def add_to_file(img_path, contains_obj, x, y, width, height):
    """ append data about a new image to the csv file """
    with open(OUTFILE_NAME, 'a') as f:
        f.write('%s,%i,%i,%i,%i,%i\n'
                % (img_path, contains_obj, x, y, width, height))
        # f.write('img,' + ','.join(lst) + '\n')
        f.close()


if __name__ == '__main__':
    root = Tk()  # root window

    if IMAGE_FOLDER is '':
        IMAGE_FOLDER = askdirectory(parent=root,
                                    title='Please select a directory')

    OUTFILE_NAME = create_initial_file(OUTFILE_NAME)

    root.attributes('-topmost', True)  # make the window pop up
    # root.geometry('300x300')
    root.geometry('%ix%i' % (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT))
    app = Window(root)  # creation of an instance
    root.mainloop()
