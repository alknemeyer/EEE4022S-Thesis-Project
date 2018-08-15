import tkinter as tk
from PIL import Image, ImageDraw

# all in pixels
img_width = 256
img_height = 256
draw_width = 5


class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.previous_x = self.previous_y = 0

        self.canvas = tk.Canvas(self, width=img_width, height=img_height,
                                bg='white', cursor='cross')
        self.canvas.pack(side='top', fill='both', expand=True)

        self.canvas.bind('<Motion>', self.tell_me_where_you_are)
        self.canvas.bind('<B1-Motion>', self.draw_from_where_you_are)
        self.canvas.bind('<Button 3>', self.save_picture)

        print('Simple Doodler')
        print('- Draw as you would normally with MS-paint')
        print('- Right click to save the image')

        self.create_new_canvas()

    def create_new_canvas(self):
        # clear canvas
        self.canvas.delete('all')

        # draw using PIL (for saving purposes only)
        self.PIL_image = Image.new('RGB', (img_width, img_height),
                                   (255, 255, 255))  # draw in white
        self.PIL_draw = ImageDraw.Draw(self.PIL_image)

    def clear_all(self):
        self.canvas.delete('all')

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y

    def draw_from_where_you_are(self, event):
        # canvas
        self.canvas.create_line(self.previous_x, self.previous_y,
                                event.x, event.y, fill='black',
                                width=draw_width)
        # PIL
        self.PIL_draw.line([self.previous_x, self.previous_y,
                           event.x, event.y], fill=(0, 0, 0),  # draw in black
                           width=draw_width)

        self.previous_x = event.x
        self.previous_y = event.y

    def save_picture(self, event):
        self.PIL_image.save('test.png')
        self.create_new_canvas()


if __name__ == '__main__':
    app = ExampleApp()
    app.mainloop()
