function gaussianRand() {

    this.generate = true;
    this.value0   = 0.0;
    this.value1   = 0.0;
    
    if(this.generate) {
      var x1 = 0.0;
      var x2 = 0.0;
      var w  = 0.0;

      do {
          // Math.random() gives value on range [0, 1) but
          // the Polar Form expects [-1, 1].
          x1 = (2.0 * Math.random()) - 1.0;
          x2 = (2.0 * Math.random()) - 1.0;
          w  = (x1 * x1) + (x2 * x2);
      } while(w >= 1.0);

      w = Math.sqrt((-2.0 * Math.log(w)) / w);

      this.value0 = x1 * w;
      this.value1 = x2 * w;

      result = this.value0;
  } else {
      result = this.value1;
  }

	this.generate = !this.generate
  return result;
}

function perform() {
    
    //Fill Array With Values
    noise = [[]];
    
    //4096 values between -1 and 1, uniform
    for(i = 0; i < 256; i++) {
        
        noise[i] = [];
        
        for(j = 0; j < 64; j++) {
            noise[i][j] = gaussianRand();
        }
    }
    
    //Convert to tfjs tensor
    inp = tf.tensor(noise)
    
    //Get Output
    if(model) {
        output = model.predict(inp).dataSync();
    } else {
        return;
    }
    
    //Make Image
    drawn = [];
    
    for(i = 0; i < 512; i++) {
        drawn[i] = []
        for(j = 0; j < 512; j++) {
            drawn[i][j] = [0, 0, 0]
        }
    }
    
    index = 0;
    
    for(im = 0; im < 256; im++) {
        for(row = 0; row < 32; row++) {
            
            onrow = row + ((im % 16) * 32)
            
            for(col = 0; col < 32; col++) {
                
                oncol = col + (Math.floor(im / 16) * 32)
                
                for(chan = 0; chan < 3; chan++) {
                    
                    //Fill Value With Value Of output[index]
                    drawn[onrow][oncol][chan] = output[index];
                    index++;
                    
                }
            }
        }
    }
    
    tf.toPixels(tf.tensor(drawn), c);
    
    
}

document.getElementById("refresh").onclick = perform;