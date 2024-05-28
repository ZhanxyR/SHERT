# Template from ControlNet https://github.com/lllyasviel/ControlNet

import gradio as gr
import numpy as np
import zerorpc
from gradio_toggle import Toggle
import open3d as o3d
from PIL import Image
import os
from tqdm import tqdm
import socket
import argparse


def process(input_image, toggle, input_mask, prompt, n_prompt, num_samples, ddim_steps, scale, control_scale, seed):

    if input_image is not None:
        input_image = input_image.reshape(args.size,args.size,3)
        image_bytes = input_image.tobytes()
    else:
        raise gr.Error('Need the input image for inpainting!')
        return

    if input_mask is not None:
        input_mask = input_mask.reshape(args.size,args.size,3)
        mask_bytes = input_mask.tobytes()
    else:
        if toggle:
            raise gr.Error('Need the input mask for local inpainting!')
            return

        mask_bytes = None

    # connection test
    def check_port(host, port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((host, port))
            s.close()
            return True
        except (socket.timeout, ConnectionRefusedError):
            return False

    if not check_port(args.host, args.port):
        raise gr.Error(f'{args.host}:{args.port} is closed.')
        return

    c = zerorpc.Client(heartbeat=None, timeout=None)
    c.connect(f'tcp://{args.host}:{args.port}')

    output = c.inpaint(image_bytes, mask_bytes, num_samples, prompt, n_prompt, ddim_steps, scale, control_scale, seed, args.size)
    output = np.frombuffer(output, dtype=np.uint8)
    output = output.reshape((num_samples, args.size, args.size, 3))

    c.close()

    return output

def toggle_action(value):

    c = zerorpc.Client(heartbeat=None, timeout=None)
    c.connect(f'tcp://{args.host}:{args.port}')

    if value:
        solver = c.switch_model('local')
    else:
        solver = c.switch_model('global')

    c.close()

def generate_mask(mask):

    mask = np.asarray(mask['layers'])
    mask_alpha = mask[...,3]
    mask = mask[...,:3]
    mask[mask_alpha==0]=0

    return mask
    
def show_render(mesh_path, texture):

    verts = o3d.io.read_triangle_mesh(mesh_path)
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.vertices = verts.vertices
    texture = np.flipud(texture)
    texture = Image.fromarray(texture)
    # texture.save('temp/texture.png')
    # texture = Image.open('temp/texture.png')
    mesh.textures = [o3d.geometry.Image(np.asarray(texture))]

    if not os.path.exists('temp'):
        os.makedirs('temp')

    imgs = []

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920,height=1080,visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().background_color = (0.8, 0.8, 0.8)
    meshShadeOption = o3d.visualization.MeshShadeOption(5)
    vis.get_render_option().light_on = True
    vis.get_render_option().mesh_show_wireframe = False
    vis.get_render_option().mesh_show_back_face = False

    render_num = 8

    for i in tqdm(range(render_num)):

        # rotation in axis-y
        angle = i * np.pi * 2 / render_num
        R = np.asarray([[np.cos(angle), 0, np.sin(angle), 0],
                    [0, 1, 0, 0],
                    [-np.sin(angle), 0, np.cos(angle), 0],
                    [0, 0, 0, 1]])
        mesh.rotate(R[:3,:3], mesh.get_center())

        # rotation in axis-x
        angle = np.pi / 9 
        R = np.asarray([[1, 0, 0, 0],
                        [0, np.cos(angle), -np.sin(angle), 0],
                        [0, np.sin(angle), np.cos(angle), 0],
                        [0, 0, 0, 1]])
        mesh.rotate(R[:3, :3], mesh.get_center())

        vis.update_geometry(mesh)
        vis.capture_screen_image(f'temp/render_view_{i}_.png', do_render=True)

        # rotate back
        angle = -np.pi / 9  
        R = np.asarray([[1, 0, 0, 0],
                        [0, np.cos(angle), -np.sin(angle), 0],
                        [0, np.sin(angle), np.cos(angle), 0],
                        [0, 0, 0, 1]])
        mesh.rotate(R[:3, :3], mesh.get_center())

        image = Image.open(f'temp/render_view_{i}_.png')

        imgs.append(image)

    return imgs


# https://github.com/gradio-app/gradio/issues/7384
# force dark theme
js_func = '''
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
'''

def parse():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--host', type=str, default='0.0.0.0', required=True, help='Server IP')
    parser.add_argument('-p', '--port', type=int, default=4242)
    parser.add_argument('-s', '--size', type=int, default=1024)
    parser.add_argument('-m', '--mesh', type=str, default='data/smplx/vt_example.obj', help='Provide the vt information')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse()

    block = gr.Blocks(js=js_func).queue()
    with block:
        with gr.Row():
            gr.Markdown('## Texture Diffusion of SHERT')
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Accordion('Inputs', open=True):
                        with gr.Row():
                            input_image = gr.Image(sources='upload', type='numpy')
                            input_mask = gr.Image(label='Mask (for local inpainting)', sources='upload', type='numpy')
                        with gr.Row():
                            with gr.Column():
                                toggle = Toggle(label='Use local Inpainting', value=True, show_label=True, info='Activate local mode and use mask image.')
                                prompt = gr.Textbox(label='Prompt', value='')
                                run_button = gr.Button(value='Run')
                            with gr.Column():
                                loader = gr.Model3D(label='Mesh Input')
                                mesh_button = gr.Button(value='Render')
                    # with gr.Row():
                    with gr.Accordion('Edited Masks', open=True):
                        mask_gallery = gr.Gallery(label='', elem_id='gallery' )
                with gr.Column():
                    with gr.Accordion('Outputs', open=True):
                        result_gallery = gr.Gallery(label='Inpainting Results', elem_id='gallery')
                    with gr.Accordion('Textured Results', open=True):
                        viewer = gr.Gallery(label='', elem_id='gallery' )
                    with gr.Accordion('Image Holder', open=False):
                        with gr.Row():
                            image_Holder1 = gr.Gallery(label='Image Holder1', show_label=False, elem_id='gallery' )
                            image_Holder2 = gr.Gallery(label='Image Holder2', show_label=False, elem_id='gallery' )
                        with gr.Row():
                            image_Holder3 = gr.Gallery(label='Image Holder3', show_label=False, elem_id='gallery' )
                            image_Holder4 = gr.Gallery(label='Image Holder4', show_label=False, elem_id='gallery' )
                    # mesh_viewer = gr.Model3D(label='Mesh Output')
            with gr.Row():
                with gr.Accordion('Mask Editor', open=True):
                    mask_editor = gr.ImageMask(label='', brush=gr.Brush(colors=['#FFFFFF'], color_mode='fixed'))
                with gr.Accordion('Advanced options', open=True):
                    num_samples = gr.Slider(label='Samples', minimum=1, maximum=12, value=1, step=1)
                    ddim_steps = gr.Slider(label='Steps', minimum=1, maximum=100, value=30, step=1)
                    scale = gr.Slider(label='Guidance Scale', minimum=0.1, maximum=30.0, value=8.0, step=0.1)
                    control_scale = gr.Slider(label='Control Scale', minimum=0.1, maximum=1, value=0.7, step=0.1)
                    seed = gr.Slider(label='Seed', minimum=-1, maximum=2147483647, value=2048, step=1)
                    n_prompt = gr.Textbox(label='Negative Prompt', value='red, dirty, blur, messy')

        ips = [input_image, toggle, input_mask, prompt, n_prompt, num_samples, ddim_steps, scale, control_scale, seed]
        run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
        mask_editor.change(generate_mask, outputs=[mask_gallery], inputs=mask_editor)
        toggle.change(fn=toggle_action, inputs=toggle)
        mesh_button.click(fn=show_render, inputs=[loader, input_image], outputs=[viewer])

    block.launch(server_name='localhost')
    # block.launch(server_name='0.0.0.0')
