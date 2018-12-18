var model;
tf.loadModel('http://matchue.ca/p/swordgan/model1/model.json').then((x) => {
    model = x;
    console.log(model)
    perform()
});