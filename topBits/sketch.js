class ProbabilisticTop3D {
  constructor(x, y, z, bias = 0.5) {
    this.pos = createVector(x, y, z);
    this.bias = bias;
    
    // Initial orientation
    this.rotation = createVector(0, 0, random(TWO_PI));
    this.angularVel = createVector(0, 0, 0);
    this.spinning = false;
    
    // Tilt parameters
    this.tilt = 0;
    this.tiltVel = 0;
    this.tiltDirection = 0;
    this.xTilt = 0;
    this.zTilt = 0;
    
    // Precession
    this.precessionAngle = 0;
    this.precessionRate = 0;
    
    // Random destabilization point
    this.destabilizeSpeed = random(0.1, 0.15);
    this.initialTilt = random(0.05, 0.15);
    
    // Dimensions
    this.bodyHeight = 40;
    this.bodyWidth = 25;
    this.stemHeight = 15;
    this.stemRadius = 3;
    this.pointHeight = 15;
    
    this.debugMode = true;
    
    // Add history arrays for plotting
    this.timeHistory = [];
    this.spinHistory = [];
    this.wobbleHistory = [];
    this.tiltHistory = [];
    this.maxHistoryLength = 200;
  }

  spin() {
    this.spinning = true;
    this.angularVel.y = random(0.3, 0.4);
    
    // Reset all tilt parameters
    this.tilt = 0;
    this.tiltVel = 0;
    this.tiltDirection = random(TWO_PI);
    this.xTilt = 0;
    this.zTilt = 0;
    
    this.precessionAngle = random(TWO_PI);
    this.precessionRate = random(0.01, 0.02);
    
    // New random destabilization parameters
    this.destabilizeSpeed = random(0.1, 0.15);
    this.initialTilt = random(0.05, 0.15);
    
    if (this.debugMode) {
      console.log('New spin with:');
      console.log('Destabilize speed:', this.destabilizeSpeed);
      console.log('Initial tilt:', this.initialTilt);
      console.log('Tilt direction:', this.tiltDirection);
    }
  }

  update() {
    if (this.spinning) {
      // Record dynamics data
      this.timeHistory.push(frameCount);
      this.spinHistory.push(this.angularVel.y);
      this.wobbleHistory.push(sin(this.precessionAngle) * 0.05);
      this.tiltHistory.push(this.tilt);
      
      if (this.timeHistory.length > this.maxHistoryLength) {
        this.timeHistory.shift();
        this.spinHistory.shift();
        this.wobbleHistory.shift();
        this.tiltHistory.shift();
      }
      
      // Basic spin and precession
      this.rotation.y += this.angularVel.y;
      this.precessionAngle += this.precessionRate;
      
      // Faster decay rates
      this.angularVel.y *= 0.995;
      this.precessionRate *= 0.995;

      // Destabilization phase
      if (this.angularVel.y < this.destabilizeSpeed && this.tilt < this.initialTilt) {
        // Faster initial tilt
        let tiltIncrease = 0.002;
        this.tilt += tiltIncrease;
        
        // Update x and z components of tilt
        this.xTilt = cos(this.tiltDirection) * this.tilt;
        this.zTilt = sin(this.tiltDirection) * this.tilt;
      }
      // Full falling phase
      else if (this.tilt >= this.initialTilt) {
        // Calculate gravitational influence that increases as top tilts more
        let gravityEffect = map(this.tilt, this.initialTilt, HALF_PI, 0.001, 0.01);
        
        // Stronger influences for more decisive fall
        let spinInfluence = sin(this.rotation.y) * 0.001;
        let precessionInfluence = sin(this.precessionAngle) * 0.001;
        
        // Add momentum based on top's "mass" (volume)
        let volume = this.bodyWidth * this.bodyWidth * this.bodyHeight;
        let momentumFactor = volume / 10000;  // Scale factor to keep values reasonable
        
        // Combine all forces with increased gravity and momentum
        this.tiltVel += (gravityEffect * momentumFactor) + spinInfluence + precessionInfluence;
        
        // Stronger damping as we get closer to horizontal to prevent bouncing
        let dampingFactor = map(this.tilt, this.initialTilt, HALF_PI, 0.995, 0.95);
        this.tiltVel *= dampingFactor;
        
        this.tilt += this.tiltVel;
        
        // Update x and z components
        this.xTilt = cos(this.tiltDirection) * this.tilt;
        this.zTilt = sin(this.tiltDirection) * this.tilt;
        
        // Stop when fallen - ensure it's very close to horizontal
        if (this.tilt > HALF_PI - 0.01) {  // Even closer to horizontal
          this.spinning = false;
          this.tilt = HALF_PI;  // Exactly horizontal
          this.xTilt = cos(this.tiltDirection) * this.tilt;
          this.zTilt = sin(this.tiltDirection) * this.tilt;
          
          let angle = (this.rotation.y % TWO_PI);
          if (angle < 0) angle += TWO_PI;
          
          if (this.debugMode) {
            console.log('Landed!');
            console.log('Final Y rotation:', angle);
            console.log('Final tilt:', this.tilt);
            console.log('Tilt direction:', this.tiltDirection);
            console.log('Final X tilt:', this.xTilt);
            console.log('Final Z tilt:', this.zTilt);
          }
        }
      }
    }
  }

  draw() {
    push();
    let heightOffset = this.bodyHeight/2 * sin(this.tilt);
    translate(this.pos.x, this.pos.y - heightOffset, this.pos.z);
    
    // Apply rotations in consistent order
    rotateY(this.rotation.y);
    if (this.spinning) {
      rotateX(sin(this.precessionAngle) * 0.05);
      rotateZ(cos(this.precessionAngle) * 0.05);
    }
    // Apply both x and z tilts
    rotateX(this.xTilt);
    rotateZ(this.zTilt);
    
    // Draw the body parts
    push();
    normalMaterial();
    
    // Main body
    push();
    translate(0, -(this.bodyHeight/2 + this.pointHeight), 0);
    box(this.bodyWidth, this.bodyHeight, this.bodyWidth);
    pop();
    
    // Stem
    push();
    translate(0, -(this.bodyHeight + this.pointHeight + this.stemHeight/2), 0);
    cylinder(this.stemRadius, this.stemHeight, 8, 1, true, true);
    pop();
    
    // Point
    push();
    translate(0, -this.pointHeight, 0);
    cone(this.bodyWidth/2, this.pointHeight);
    pop();
    
    // Draw faces with numbers only
    translate(0, -(this.bodyHeight/2 + this.pointHeight), 0);
    
    // Draw numbered faces
    for (let i = 0; i < 4; i++) {
      push();
      rotateY(i * HALF_PI);
      translate(0, 0, this.bodyWidth/2 + 2);
      fill(200);  // All faces gray for now
      plane(this.bodyWidth * 0.9, this.bodyHeight * 0.9);
      
      // Move text slightly in front of the face
      translate(0, 0, 0.5);
      fill(0);
      textSize(20);
      textAlign(CENTER, CENTER);
      textFont(defaultFont);
      text(i.toString(), 0, 0);
      pop();
    }
    
    pop();
    pop();
  }

  drawDynamicsPlot() {
    push();
    translate(-width/4, height/3);  // Position plot in bottom left
    
    // Draw axes
    stroke(0);
    line(0, 0, 200, 0);  // X axis
    line(0, -100, 0, 100);  // Y axis
    
    // Draw legend
    textAlign(LEFT);
    textSize(12);
    fill(255, 0, 0); text("Spin", 210, -80);
    fill(0, 255, 0); text("Wobble", 210, -60);
    fill(0, 0, 255); text("Tilt", 210, -40);
    
    // Plot histories
    noFill();
    
    // Spin (red)
    stroke(255, 0, 0);
    beginShape();
    for (let i = 0; i < this.spinHistory.length; i++) {
      let x = map(i, 0, this.maxHistoryLength, 0, 200);
      let y = map(this.spinHistory[i], 0, 0.15, 0, -100);
      vertex(x, y);
    }
    endShape();
    
    // Wobble (green)
    stroke(0, 255, 0);
    beginShape();
    for (let i = 0; i < this.wobbleHistory.length; i++) {
      let x = map(i, 0, this.maxHistoryLength, 0, 200);
      let y = map(this.wobbleHistory[i], -0.05, 0.05, -100, 100);
      vertex(x, y);
    }
    endShape();
    
    // Tilt (blue)
    stroke(0, 0, 255);
    beginShape();
    for (let i = 0; i < this.tiltHistory.length; i++) {
      let x = map(i, 0, this.maxHistoryLength, 0, 200);
      let y = map(this.tiltHistory[i], 0, HALF_PI, 0, 100);
      vertex(x, y);
    }
    endShape();
    
    pop();
  }
}

let spinTop;
let defaultFont;
let resetButton;
let rotateButton;
let spinButton;

function preload() {
  defaultFont = loadFont('https://cdnjs.cloudflare.com/ajax/libs/topcoat/0.8.0/font/SourceCodePro-Bold.otf');
}

function setup() {
  createCanvas(600, 400, WEBGL);
  spinTop = new ProbabilisticTop3D(0, 0, 0, 0.5);
  
  // Create buttons in 2D screen space
  resetButton = createButton('Reset');
  resetButton.position(20, 20);
  resetButton.mousePressed(resetTop);
  
  rotateButton = createButton('Rotate');
  rotateButton.position(80, 20);
  rotateButton.mousePressed(rotateTop);
  rotateButton.attribute('disabled', ''); // Start disabled
  
  spinButton = createButton('Spin');
  spinButton.position(140, 20);
  spinButton.mousePressed(startSpin);
  spinButton.attribute('disabled', ''); // Start disabled
  
  ambientLight(60);
  directionalLight(255, 255, 255, -1, 1, -1);
  textFont(defaultFont);
}

function resetTop() {
  // Reset top to initial state
  spinTop = new ProbabilisticTop3D(0, 0, 0, 0.5);
  rotateButton.removeAttribute('disabled'); // Enable rotate button
  spinButton.removeAttribute('disabled');   // Enable spin button
}

function rotateTop() {
  // Randomly rotate the top without spinning
  spinTop.rotation.y = random(TWO_PI);
  if (spinTop.debugMode) {
    console.log('Rotated to:', spinTop.rotation.y);
  }
}

function startSpin() {
  spinTop.spin();
  rotateButton.attribute('disabled', ''); // Disable rotate until next reset
  spinButton.attribute('disabled', '');   // Disable spin until next reset
}

function draw() {
  background(255);
  orbitControl();
  
  // Set initial camera angle to see two faces
  rotateX(-0.6);
  rotateY(-PI/4);
  
  // Draw ground plane
  push();
  translate(0, 0, 0);
  rotateX(HALF_PI);
  fill(240);
  plane(200, 200);
  pop();
  
  spinTop.update();
  spinTop.draw();
  spinTop.drawDynamicsPlot();  // Add the dynamics plot
}