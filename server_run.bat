@echo off
cd api
start cmd /k "nodemon server.js"

cd ..
cd web-inpaint
start cmd /k "npm start"