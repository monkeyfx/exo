from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import numpy as np
import cv2

def draw_rounded_rectangle(draw, coords, radius, fill):
  """
  绘制圆角矩形
  Args:
      draw: ImageDraw对象，用于绘图
      coords: 矩形的坐标(left, top, right, bottom)
      radius: 圆角半径
      fill: 填充颜色
  """
  left, top, right, bottom = coords
  diameter = radius * 2
  # 绘制矩形的中间部分
  draw.rectangle([left + radius, top, right - radius, bottom], fill=fill)
  draw.rectangle([left, top + radius, right, bottom - radius], fill=fill)
  # 绘制四个角的圆弧
  draw.pieslice([left, top, left + diameter, top + diameter], 180, 270, fill=fill)
  draw.pieslice([right - diameter, top, right, top + diameter], 270, 360, fill=fill)
  draw.pieslice([left, bottom - diameter, left + diameter, bottom], 90, 180, fill=fill)
  draw.pieslice([right - diameter, bottom - diameter, right, bottom], 0, 90, fill=fill)

def draw_centered_text_rounded(draw, text, font, rect_coords, radius=10, text_color="yellow", bg_color=(43,33,44)):
  """
  在圆角矩形中绘制居中文本
  Args:
      draw: ImageDraw对象
      text: 要绘制的文本
      font: 字体对象
      rect_coords: 矩形坐标
      radius: 圆角半径
      text_color: 文本颜色
      bg_color: 背景颜色
  """
  bbox = font.getbbox(text)
  text_width = bbox[2] - bbox[0]
  text_height = bbox[3] - bbox[1]
  rect_left, rect_top, rect_right, rect_bottom = rect_coords
  rect_width = rect_right - rect_left
  rect_height = rect_bottom - rect_top
  text_x = rect_left + (rect_width - text_width) // 2
  text_y = rect_top + (rect_height - text_height) // 2
  draw_rounded_rectangle(draw, rect_coords, radius, bg_color)
  draw.text((text_x, text_y), text, fill=text_color, font=font)

def draw_left_aligned_text_rounded(draw, text, font, rect_coords, padding_left=20, radius=10, text_color="yellow", bg_color=(43,33,44)):
  """
  在圆角矩形中绘制左对齐文本
  Args:
      draw: ImageDraw对象
      text: 要绘制的文本
      font: 字体对象
      rect_coords: 矩形坐标
      padding_left: 左侧填充
      radius: 圆角半径
      text_color: 文本颜色
      bg_color: 背景颜色
  """
  bbox = font.getbbox(text)
  text_height = bbox[3] - bbox[1]
  rect_left, rect_top, rect_right, rect_bottom = rect_coords
  rect_height = rect_bottom - rect_top
  text_y = rect_top + (rect_height - text_height) // 2
  text_x = rect_left + padding_left
  draw_rounded_rectangle(draw, rect_coords, radius, bg_color)
  draw.text((text_x, text_y), text, fill=text_color, font=font)

def draw_right_text_dynamic_width_rounded(draw, text, font, base_coords, padding=20, radius=10, text_color="yellow", bg_color=(43,33,44)):
  """
  在圆角矩形中绘制右对齐文本，矩形宽度动态调整
  Args:
      draw: ImageDraw对象
      text: 要绘制的文本
      font: 字体对象
      base_coords: 基础坐标
      padding: 填充
      radius: 圆角半径
      text_color: 文本颜色
      bg_color: 背景颜色
  Returns:
      新的矩形左边界
  """
  bbox = font.getbbox(text)
  text_width = bbox[2] - bbox[0]
  text_height = bbox[3] - bbox[1]
  _, rect_top, rect_right, rect_bottom = base_coords
  rect_height = rect_bottom - rect_top
  new_rect_left = rect_right - (text_width + (padding * 2))
  text_y = rect_top + (rect_height - text_height) // 2
  text_x = new_rect_left + padding
  draw_rounded_rectangle(draw, (new_rect_left, rect_top, rect_right, rect_bottom), radius, bg_color)
  draw.text((text_x, text_y), text, fill=text_color, font=font)
  return new_rect_left

def draw_progress_bar(draw, progress, coords, color="yellow", bg_color=(70, 70, 70)):
  """
  绘制进度条
  Args:
      draw: ImageDraw对象
      progress: 进度（0到1之间）
      coords: 进度条坐标
      color: 进度条颜色
      bg_color: 背景颜色
  """
  left, top, right, bottom = coords
  total_width = right - left
  draw.rectangle(coords, fill=bg_color)
  progress_width = int(total_width * progress)
  if progress_width > 0:
    draw.rectangle((left, top, left + progress_width, bottom), fill=color)

def crop_image(image, top_crop=70):
  """
  裁剪图像顶部
  Args:
      image: PIL图像对象
      top_crop: 裁剪的像素数
  Returns:
      裁剪后的图像
  """
  width, height = image.size
  return image.crop((0, top_crop, width, height))

def create_animation_mp4(
  replacement_image_path,
  output_path,
  device_name,
  prompt_text,
  fps=30,
  target_size=(512, 512),
  target_position=(139, 755),
  progress_coords=(139, 1285, 655, 1295),
  device_coords=(1240, 370, 1640, 416),
  prompt_coords=(332, 1702, 2662, 1745)
):
  """
  创建动画并保存为MP4格式
  Args:
      replacement_image_path: 替换图像路径
      output_path: 输出视频路径
      device_name: 设备名称
      prompt_text: 提示文本
      fps: 帧率
      target_size: 目标图像大小
      target_position: 目标图像位置
      progress_coords: 进度条坐标
      device_coords: 设备名称坐标
      prompt_coords: 提示文本坐标
  """
  frames = []
  try:
    # 尝试加载字体
    font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 20)
    promptfont = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 24)
  except:
    # 如果加载失败，使用默认字体
    font = ImageFont.load_default()
    promptfont = ImageFont.load_default()

  # 处理第一帧
  base_img = Image.open(os.path.join(os.path.dirname(__file__), "baseimages", "image1.png"))
  draw = ImageDraw.Draw(base_img)
  draw_centered_text_rounded(draw, device_name, font, device_coords)
  frames.extend([crop_image(base_img)] * 30)  # 1秒钟30帧

  # 处理第二帧，带有打字动画
  base_img2 = Image.open(os.path.join(os.path.dirname(__file__), "baseimages", "image2.png"))
  for i in range(len(prompt_text) + 1):
    current_frame = base_img2.copy()
    draw = ImageDraw.Draw(current_frame)
    draw_centered_text_rounded(draw, device_name, font, device_coords)
    if i > 0:  # 只有在有至少一个字符时才绘制
      draw_left_aligned_text_rounded(draw, prompt_text[:i], promptfont, prompt_coords)
    frames.extend([crop_image(current_frame)] * 2)  # 每个字符2帧，平滑打字效果
  
  # 保持完整提示一段时间
  frames.extend([frames[-1]] * 30)  # 保持1秒

  # 创建模糊序列
  replacement_img = Image.open(replacement_image_path)
  base_img = Image.open(os.path.join(os.path.dirname(__file__), "baseimages", "image3.png"))
  blur_steps = [int(80 * (1 - i/8)) for i in range(9)]

  for i, blur_amount in enumerate(blur_steps):
    new_frame = base_img.copy()
    draw = ImageDraw.Draw(new_frame)

    replacement_copy = replacement_img.copy()
    replacement_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
    if blur_amount > 0:
      replacement_copy = replacement_copy.filter(ImageFilter.GaussianBlur(radius=blur_amount))

    mask = replacement_copy.split()[-1] if replacement_copy.mode in ('RGBA', 'LA') else None
    new_frame.paste(replacement_copy, target_position, mask)

    draw_progress_bar(draw, (i + 1) / 9, progress_coords)
    draw_centered_text_rounded(draw, device_name, font, device_coords)
    draw_right_text_dynamic_width_rounded(draw, prompt_text, promptfont, (None, 590, 2850, 685), padding=30)

    frames.extend([crop_image(new_frame)] * 15)  # 0.5秒钟30帧

  # 创建并添加最终帧（image4）
  final_base = Image.open(os.path.join(os.path.dirname(__file__), "baseimages", "image4.png"))
  draw = ImageDraw.Draw(final_base)

  draw_centered_text_rounded(draw, device_name, font, device_coords)
  draw_right_text_dynamic_width_rounded(draw, prompt_text, promptfont, (None, 590, 2850, 685), padding=30)

  replacement_copy = replacement_img.copy()
  replacement_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
  mask = replacement_copy.split()[-1] if replacement_copy.mode in ('RGBA', 'LA') else None
  final_base.paste(replacement_copy, target_position, mask)

  frames.extend([crop_image(final_base)] * 30)  # 1秒钟30帧

  # 使用H.264编解码器将帧转换为视频
  if frames:
    first_frame = np.array(frames[0])
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(
      output_path,
      fourcc,
      fps,
      (width, height),
      isColor=True
    )

    if not out.isOpened():
      print("Error: VideoWriter failed to open")
      return

    for frame in frames:
      frame_array = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
      out.write(frame_array)
    
    out.release()
    print(f"Video saved successfully to {output_path}")