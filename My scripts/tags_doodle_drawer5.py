import tkinter as tk
from tkinter.filedialog import askdirectory
from PIL import Image, ImageDraw, ImageTk
import os
from pathlib import Path
import argparse


class SideBySideDoodler(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master  # master widget = tk window
        self.master.title('Simple Doodle Drawer')
        self.pack(fill=tk.BOTH, expand=1)  # give widget full space of root window

        self.previous_x = self.previous_y = 0

        # sort out menu stuff
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        file = tk.Menu(menu)
        file.add_command(label='Clear canvas', command=self.create_new_canvas)
        file.add_command(label='Save picture', command=self.save_picture,
                         accelerator='Right click')
        menu.add_cascade(label='File', menu=file)

        # create the canvas which we'll draw on
        self.canvas = tk.Canvas(self, width=img_width*2, height=img_height,
                                bg='white', cursor='cross')
        self.canvas.pack(side='top', fill='both', expand=True)

        # bind motions of the mouse
        self.canvas.bind('<Motion>', self.tell_me_where_you_are)
        self.canvas.bind('<B1-Motion>', self.draw_from_where_you_are)
        self.canvas.bind('<Button 3>', self.save_picture_mouse_click)

        # do directory stuff with pics
        self.input_dir = Path(askdirectory(
            parent=master,
            title='Please select a directory'))
        self.dirs = [Path(dir) for dir in os.listdir(self.input_dir)]
        self.index = 0

        self.create_new_canvas()
        self.show_new_picture()

    def create_new_canvas(self):
        # clear canvas
        self.canvas.delete('all')

        # draw using PIL (for saving purposes only)
        # use twice the image width for canvas + img you're copying
        self.PIL_image = Image.new('RGB', (img_width*2, img_height),
                                   (255, 255, 255, 255))  # draw in white
        self.PIL_draw = ImageDraw.Draw(self.PIL_image)

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x % img_width
        self.previous_y = event.y

    def draw_from_where_you_are(self, event):
        # canvas
        print(f'x, y = {event.x}, {event.y}')
        self.canvas.create_line(self.previous_x, self.previous_y,
                                event.x % img_width, event.y,
                                fill='black', width=draw_width)
        # PIL
        self.PIL_draw.line([self.previous_x, self.previous_y,
                           event.x % img_width, event.y],
                           fill=(0, 0, 0),  # draw in black
                           width=draw_width)

        self.previous_x = event.x % img_width
        self.previous_y = event.y

    def save_picture_mouse_click(self, event):
        self.save_picture()

    def save_picture(self):
        self.PIL_image.paste(self.load, (img_width, 0))

        self.PIL_image.save(output_dir/self.dirs[self.index-1],
                            format='png')

        print(f'Saved as {self.dirs[self.index-1]}')

        if self.index >= len(self.dirs):
            quit()

        self.create_new_canvas()
        self.show_new_picture()

    def show_new_picture(self):
        while self.index < len(self.dirs):
            next_img = self.dirs[self.index]
            if next_img not in os.listdir(output_dir):
                try:
                    self.load = Image.open(self.input_dir/next_img)
                    self.index += 1
                    break
                except:
                    print(f'Skipping file {next_img}')
                    self.index += 1

        if self.index >= len(self.dirs):
            quit()

        self.load = self.load.resize((img_width, img_height),
                                     Image.ANTIALIAS)
        render = ImageTk.PhotoImage(self.load)

        # labels can be text or images
        img = tk.Label(self, image=render)
        img.image = render
        img.place(x=img_width, y=0)

        self.index += 1

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Simple Doodle Drawer')
    p.add_argument('-W', '--img_width', default=256,
                   help='width in pixels of the output image (default: 256)')
    p.add_argument('-H', '--img_height', default=256,
                   help='height in pixels of the output image (default: 256)')
    p.add_argument('-D', '--draw_width', default=5,
                   help='width of the line to be drawn (default: 5 pix)')
    p.add_argument('-O', '--output_dir', default='image_out',
                   help='name of output image directory (default: image_out)')

    args = p.parse_args()
    img_width = args.img_width
    img_height = args.img_height
    draw_width = args.draw_width
    output_dir = Path(args.output_dir)

    print('------------------------------------------')
    print('Simple Doodler')
    print('- Draw as you would normally with MS-paint')
    print('- Right click to save the image')
    print('- Don\'t resize the window! It\'ll mess things up!')
    print(f'- The images will be saved in "{output_dir}", \
found in the same directory as the command prompt which is launching is')

    print(f'\nYour current directory is {os.getcwd()}')

    if output_dir.as_posix() not in os.listdir():
        os.mkdir(output_dir.as_posix())

    root = tk.Tk()
    root.attributes('-topmost', True)  # make the window pop up
    app = SideBySideDoodler(root)
    app.mainloop()
