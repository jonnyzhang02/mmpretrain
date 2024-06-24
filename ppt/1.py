# from PIL import Image
# import os

# def split_image(image_path, rows, cols, save_dir):
#     # 加载图片
#     img = Image.open(image_path)
#     w, h = img.size
    
#     # 计算每块的尺寸
#     block_w, block_h = w // cols, h // rows
    
#     # 创建保存目录
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 按照行和列拆分图片
#     for i in range(rows):
#         for j in range(cols):
#             # 计算当前块的坐标
#             left, top = j * block_w, i * block_h
#             right, bottom = left + block_w, top + block_h
            
#             # 裁剪图片
#             block = img.crop((left, top, right, bottom))
            
#             # 保存图片
#             block.save(os.path.join(save_dir, f'block_{i+1}_{j+1}.png'))

# # 使用示例
# image_path = '1.png'  # 需要拆分的图片路径
# save_dir = 'blocks'       # 保存拆分后的图片的文件夹路径
# split_image(image_path, 5, 5, save_dir)
from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(video_filepath, output_filepath, start_time=0, end_time=None, resize=None):
    """
    将MP4视频转换为GIF。

    :param video_filepath: 输入的视频文件路径。
    :param output_filepath: 输出GIF的文件路径。
    :param start_time: 视频起始时间，单位为秒。
    :param end_time: 视频结束时间，单位为秒。
    :param resize: 调整GIF的大小，例如(320, 240)。
    """
    # 加载视频文件
    clip = VideoFileClip(video_filepath)
    
    # 如果指定了结束时间，截取视频的一部分
    if end_time:
        clip = clip.subclip(start_time, end_time)
    else:
        clip = clip.subclip(start_time)

    # 如果指定了新的大小，调整视频大小
    if resize:
        clip = clip.resize(resize)
    
    # 转换为GIF并保存
    clip.write_gif(output_filepath, fps=30)  # 可以调整fps来改变GIF的帧率

# 使用示例
convert_mp4_to_gif('2.mp4', 'output2.gif')


