import app

#message = "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
message = "Free 1st week entry 2 TEXTPOD 4 a chance 2 win 40GB iPod or �250 cash every wk. Txt POD to 84128 Ts&Cs www.textpod.net"

result = app.predict(message)

print(app.predict(message))
print(app.predict_proba(message))