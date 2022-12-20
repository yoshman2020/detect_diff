import base64
import os
from bottle import get, post, request, run

import detectdiff


@get('/')
def index():
    return '''
        <form action="/" method="post" enctype="multipart/form-data">
            Select a file1: <input type="file" name="file1" />
            Select a file2: <input type="file" name="file2" />
            <input type="submit" value="Start" />
        </form>
    '''


@post('/')
def do_ditect_diff():
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')
    name, ext = os.path.splitext(file1.filename)
    if ext not in ('.png', '.jpg', '.jpeg'):
        return 'File extension not allowed.'
    name, ext = os.path.splitext(file2.filename)
    if ext not in ('.png', '.jpg', '.jpeg'):
        return 'File extension not allowed.'

    path_a = 'tempA'
    path_b = 'tempB'

    if os.path.exists(path_a):
        os.remove(path_a)
    if os.path.exists(path_b):
        os.remove(path_b)
    file1.save(path_a)
    file2.save(path_b)

    result = detectdiff.ditect_diff(path_a, path_b, ext)

    if result != '':
        data = file2.file.read()
        file2_base64 = base64.b64encode(data).decode('utf-8')
        return f'''
            <div display='flex'>
                <img src="data:image/png;base64,{result}" />
                <img src="data:image/png;base64,{file2_base64}" />
                <br />
                <a href="/">back</a>
            </div>
        '''
    return 'error!'


def main():
    run(host='0.0.0.0', port=58131)


if __name__ == "__main__":
    main()
