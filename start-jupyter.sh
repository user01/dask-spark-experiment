openssl req -new -newkey rsa:2048 -days 365 -nodes -x509 -subj '/C=XX/ST=XX/L=XX/O=generated/CN=generated' -keyout /local_cert.pem -out /local_cert.pem
chmod 444 /local_cert.pem
echo "
c = get_config()
c.NotebookApp.password = '$(python -c "from notebook.auth import passwd; print(passwd(open('/root/.jupyter/password', 'r').read()), end='')")'
" > /root/.jupyter/jupyter_notebook_config.py
jupyter lab --no-browser --allow-root --ip=0.0.0.0 --certfile=/local_cert.pem