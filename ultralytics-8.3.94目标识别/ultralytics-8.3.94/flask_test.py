from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    # 模拟数据
    data = {'message': 'Hello, Flask!'}
    # 简单的 HTML 模板
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Flask App</title>
    </head>
    <body>
        <h1>{{ message }}</h1>
    </body>
    </html>
    """
    return render_template_string(template, **data)

if __name__ == '__main__':
    app.run(debug=True)