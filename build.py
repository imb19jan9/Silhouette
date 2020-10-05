import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import glm

from stl import mesh
from PIL import Image

import numpy as np
from math import *
import random
import os
import csv


def toMat3(q):
    # quaternion to 3x3 matrix
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    mat = glm.mat3(1.0)

    w, x, y, z = q[0], q[1], q[2], q[3]
    qq = q * q
    sqw, sqx, sqy, sqz = qq[0], qq[1], qq[2], qq[3]

    mat[0, 0] = sqx - sqy - sqz + sqw
    mat[1, 1] = -sqx + sqy - sqz + sqw
    mat[2, 2] = -sqx - sqy + sqz + sqw

    tmp1 = x * y
    tmp2 = z * w
    mat[1, 0] = 2 * (tmp1 + tmp2)
    mat[0, 1] = 2 * (tmp1 - tmp2)

    tmp1 = x * z
    tmp2 = y * w
    mat[2, 0] = 2 * (tmp1 - tmp2)
    mat[0, 2] = 2 * (tmp1 + tmp2)

    tmp1 = y * z
    tmp2 = x * w
    mat[2, 1] = 2 * (tmp1 + tmp2)
    mat[1, 2] = 2 * (tmp1 - tmp2)

    return mat


def toMat4(q):
    # quaternion to 4x4 matrix
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    mat = glm.mat4(1.0)

    w, x, y, z = q[0], q[1], q[2], q[3]
    qq = q * q
    sqw, sqx, sqy, sqz = qq[0], qq[1], qq[2], qq[3]

    mat[0, 0] = sqx - sqy - sqz + sqw
    mat[1, 1] = -sqx + sqy - sqz + sqw
    mat[2, 2] = -sqx - sqy + sqz + sqw

    tmp1 = x * y
    tmp2 = z * w
    mat[1, 0] = 2 * (tmp1 + tmp2)
    mat[0, 1] = 2 * (tmp1 - tmp2)

    tmp1 = x * z
    tmp2 = y * w
    mat[2, 0] = 2 * (tmp1 - tmp2)
    mat[0, 2] = 2 * (tmp1 + tmp2)

    tmp1 = y * z
    tmp2 = x * w
    mat[2, 1] = 2 * (tmp1 + tmp2)
    mat[1, 2] = 2 * (tmp1 - tmp2)

    return mat


def bbox_info(points):
    bbox_max = np.max(points, axis=0)
    bbox_min = np.min(points, axis=0)
    bbox_center = (bbox_max + bbox_min) / 2.0
    bbox_size = bbox_max - bbox_min

    return bbox_max, bbox_min, bbox_center, bbox_size


def random_uniform_rotation():
    # random quaternion
    # http://planning.cs.uiuc.edu/node198.html
    u1, u2, u3 = [random.random() for i in range(3)]
    sigma1, sigma2 = sqrt(1 - u1), sqrt(u1)
    theta1, theta2 = 2 * pi * u2, 2 * pi * u3
    h = np.array(
        [
            cos(theta2) * sigma2,
            sin(theta1) * sigma1,
            cos(theta1) * sigma1,
            sin(theta2) * sigma2,
        ]
    )

    return toMat3(h)


def glfw_initialize():
    # initialize glfw
    if not glfw.init():
        return

    win_width, win_height = 100, 100
    window = glfw.create_window(win_width, win_height, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    return window


def setup_fbo(fbo_width, fbo_height):
    # fbo
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    # fbo color texture
    color_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, color_tex)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA, fbo_width, fbo_height, 0, GL_RGBA, GL_FLOAT, 0
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    # fbo depth texture
    depth_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_tex)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_DEPTH_COMPONENT,
        fbo_width,
        fbo_height,
        0,
        GL_DEPTH_COMPONENT,
        GL_FLOAT,
        0,
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color_tex, 0)
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_tex, 0)
    draw_buffers = [GL_COLOR_ATTACHMENT0]
    glDrawBuffers(1, draw_buffers)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return fbo


def setup_shader():
    # shader source
    vertex_shader = """
    #version 430
    layout(location = 0) in vec3 position;
    uniform mat4 mvp;
    void main()
    {
        gl_Position = mvp * vec4(position, 1.0f);
    }
    """

    fragment_shader = """
    #version 430
    out vec4 outColor;
    void main()
    {
        outColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    """
    shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )

    mvploc = glGetUniformLocation(shader, "mvp")

    return shader, mvploc


def get_mvp(scale):
    # camera configuration
    camPos = glm.vec3(0.0, 0.0, -5e7)
    camTarget = glm.vec3(0.0, 0.0, 0.0)
    camDir = glm.normalize(camPos - camTarget)
    camUp = glm.vec3(0.0, 1.0, 0.0)
    camRight = glm.normalize(glm.cross(camUp, camDir))
    camUp = glm.cross(camDir, camRight)

    # model view projecction matrix
    view = glm.lookAt(camPos, camTarget, camUp)
    projection = glm.ortho(-1.0, 1.0, -1.0, 1.0, 0.1, 1e8)
    model = glm.scale(glm.mat4(1.0), glm.vec3(scale))
    mvp = projection * view * model

    return mvp


def setup_glBuffer(m, rotation):
    points = m.points.reshape(-1, 3)

    # random rotation
    points = np.matmul(rotation, points.T).T

    # bounding box centering
    _, _, bbox_center, bbox_size = bbox_info(points)
    points -= bbox_center

    # scale to set the bounding box's size withn (-1,1)
    bbox_target = 2
    scale = (
        bbox_target / bbox_size[0]
        if bbox_size[0] > bbox_size[1]
        else bbox_target / bbox_size[1]
    )
    points *= scale

    # flatten point array
    points = points.flatten().astype(np.float32)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(
        GL_ARRAY_BUFFER,
        points.size * ctypes.sizeof(ctypes.c_float),
        points,
        GL_STATIC_DRAW,
    )

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)

    return vao, vbo


def main():
    window = glfw_initialize()

    fbo_width, fbo_height = 50, 50
    fbo = setup_fbo(fbo_width, fbo_height)

    shader, mvploc = setup_shader()

    num_view = 20
    random_rotation = [random_uniform_rotation() for _ in range(num_view)]

    f = open("Thingi10K/goodfile_id.csv", 'r', encoding='utf-8')
    rdr = csv.reader(f)
    filenames = ["Thingi10K/raw_meshes/"+line[0]+".stl" for line in rdr]
    f.close() 

    for idx, filename in enumerate(filenames):
        print(idx)
        if not os.path.isfile(filename):
            continue
        
        # read mesh
        m = mesh.Mesh.from_file(filename)

        for scale in [0.3, 0.6, 0.9]:
            for i in range(num_view):
                vao, vbo = setup_glBuffer(m, random_rotation[i])
                mvp = get_mvp(scale)

                glUseProgram(shader)
                glUniformMatrix4fv(mvploc, 1, GL_FALSE, glm.value_ptr(mvp))

                glBindFramebuffer(GL_FRAMEBUFFER, fbo)
                glViewport(0, 0, fbo_width, fbo_height)
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT)

                glBindVertexArray(vao)
                glDrawArrays(GL_TRIANGLES, 0, m.v0.shape[0] * 3)

                glfw.swap_buffers(window)
                glfw.poll_events()

                captured = glReadPixels(0, 0, fbo_width, fbo_height, GL_RGB, GL_FLOAT)
                captured = (captured * 255).astype(np.uint8)
                img = Image.fromarray(captured)
                file_id = os.path.splitext(filename)[0].split('/')[-1]
                img.save(f"size50/{file_id}_sc{scale}_view{i}.png")
                glBindFramebuffer(GL_FRAMEBUFFER, 0)

                glDeleteBuffers(1, [vbo])
                glDeleteVertexArrays(1, [vao])

    glfw.terminate()


if __name__ == "__main__":
    main()