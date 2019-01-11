var express = require("express"),
    app = express(),
    bodyParser = require('body-parser'),
    Images = require('./routes/Images'),
    port = process.env.PORT || 3001;

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));

app.use('/api/backend/Images', Images);

app.listen(port, function(){
    console.log("Server is running on port " + port);
})
