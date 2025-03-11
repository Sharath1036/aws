pip install fastapi uvicorn mangum

pip freeze > requirements.txt (only need fastapi and mangum)

create main.py

'''
from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

@app.get("/")
async def hello():
  return {"message": "hello world"}
'''

uvicorn main:app --reload

commands to be deployed on aws lamba

pip3 install -t dependencies -r requirements.txt ( a dependencies forlder is formed)

(cd dependencies; zip ../aws_lambda_artifcat.zip -r .)

zip aws_lambda_artifcat.zip -u main.py

head over to aws console
type lambda
create function
function name: any
runtime:

advance senttings: enable function url (to assign https endpoints to lambda functions)
auth type: none

create function

we get a function url
status code: 200 is the response

upload a .zip file (artifact earlier)
click edit under runtime settings

handler: main



