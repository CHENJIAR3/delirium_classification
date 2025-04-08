# -*- coding: utf-8 -*-
# 导入库函数
import numpy as np
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash,session,send_from_directory,send_file
import logging
import torch
import os
from app_function import initialize_model,load_fixed_frames,load_sliding_frames
from werkzeug.utils import secure_filename
import sys
from io import BytesIO
from utils.experimental_setting import args
# Flask初始化
app = Flask(__name__)


app.logger.setLevel(logging.INFO)

app.secret_key = 'for_children'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['Result_dir'] = "static/results"
app.config["file_data"] = bytes([])



def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
def clear_session():
    """清除 session 中的视频和分类结果"""
    if 'last_video' in session:
        del session['last_video']
    if 'classification_result' in session:
        del session['classification_result']

def delete_uploaded_file(filepath):
    """删除上传的文件"""
    if os.path.exists(filepath):
        os.remove(filepath)
        app.logger.info(f"Deleted file: {filepath}")


@app.route('/', methods=['GET', 'POST'])
def index():

    video_url = session.get('last_video')  # 从session获取最近上传视频
    classification_result = session.get('classification_result')
    prediction = [0.00,0.00]

    if request.method == 'POST':
        model_type = request.form.get('model_type')
        args.model_type= model_type
        print(request.form.get('model_type'))
        print(f"[DEBUG] 选择的模型类型: {model_type}")
        if model_type is None: model_type = "mc3_18"
        model, device,transform = initialize_model(args)
        print(model_type)

        session['model_type'] = model_type

        if 'file' not in request.files:
            flash('请选择合适的文件')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('请选择合适的文件')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            flash(f'当前模型选择为{model_type}，文件上传成功: {file.filename}', 'warning')

            file_data = file.read()
            app.config["file_data"] = file_data
            file.seek(0)

            # 清除 session 中的旧数据
            clear_session()
            # video_url = file.filename
            video_url = secure_filename(file.filename)
            upload_dir = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            if not os.path.exists(app.config['Result_dir']):
                os.makedirs(app.config['Result_dir'])
            # 处理视频帧并进行分类
            try:
                filepath = os.path.join(upload_dir, video_url)
                try:
                    file.save(filepath)
                    session['last_video'] = video_url  # 存储到session
                except Exception as e:
                    flash(f'文件上传失败: {str(e)}')
                frames = load_sliding_frames(filepath, max_num=10,transform=transform).to(device)
                if frames.ndim < 5:
                    frames = frames.unsqueeze(0)
                with torch.no_grad():
                    prediction = model(frames).detach().cpu().numpy()
                prediction = np.round(np.sum(prediction, axis=0) / np.sum(prediction), 4)
                classification_result = '阳性' if np.argmax(prediction) == 0 else '阴性'
                session['classification_result'] = classification_result
                resultpath = os.path.join(app.config['Result_dir'], video_url.split('.')[0]+"模型预测为"+classification_result+".mp4")
                # 指针重置到开头
                file.seek(0)
                # 保存为结果文件
                file.save(resultpath)

                final_diagnosis = request.get_json.get('final_diagnosis')

                print(final_diagnosis)
            except Exception as e:
                flash(f'视频处理失败: {str(e)}')
                delete_uploaded_file(filepath)
                return redirect(request.url)

            print(prediction)
            return render_template('index.html', video_url=video_url, classification_result=classification_result,
                                   prediction=prediction)
            # 删除上传的文件以释放磁盘空间
            delete_uploaded_file(filepath)
        else:
            flash('仅支持MP4格式')

    return render_template('index.html', video_url=video_url, classification_result=classification_result,
                                   prediction=prediction)
# 提供视频文件的直接访问（通过路由保护）
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(
        BytesIO(app.config["file_data"] ),   # 内存文件对象
        mimetype='video/mp4',
        download_name='video.mp4'
    )
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)