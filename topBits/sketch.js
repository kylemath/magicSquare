class SpinningTop {
  constructor(x, y, z) {
    // Position - set y to bodyHeight/2 so bottom touches floor
    this.pos = createVector(x, this.bodyHeight/2, z);
    
    // Dimensions
    this.bodyHeight = 40;
    this.bodyWidth = 25;
    
    // Physical properties
    this.angularVel = 0;      // Angular velocity around y-axis
    this.tilt = 0;            // Tilt angle from vertical
    this.tiltDir = 0;         // Direction of tilt
    this.rotation = 0;        // Current rotation angle
    this.tiltVelocity = 0;    // Rate of change of tilt
    
    // Physics constants
    this.gravity = 0.098;     // Gravitational acceleration
    this.momentOfInertia = 1000;  // Resistance to rotation
    
    // State
    this.spinning = false;
    this.falling = false;
    this.currentFace = 0;
    
    // Energy loss coefficients
    this.spinDamping = 0.997;
    this.tiltDamping = 0.995;
    
    // Critical thresholds
    this.minSpinSpeed = 0.5;
    this.criticalTilt = PI/4;  // 45 degrees
  }

  spin() {
    this.spinning = true;
    this.falling = false;
    
    // Random initial conditions
    this.angularVel = random(30, 40);
    this.tilt = random(0.05, 0.15);
    this.tiltDir = random(TWO_PI);
    this.rotation = random(TWO_PI);
    this.tiltVelocity = 0;
  }

  update() {
    if (!this.spinning) return;

    if (!this.falling) {
      // Update rotation based on angular velocity
      this.rotation += this.angularVel * 0.01;
      
      // Calculate gyroscopic effect (resistance to tilting)
      let gyroscopicTorque = (this.angularVel * this.angularVel) / this.momentOfInertia;
      
      // Calculate gravitational torque
      let gravityTorque = this.gravity * sin(this.tilt);
      
      // Net acceleration of tilt
      let tiltAccel = gravityTorque - gyroscopicTorque * sin(this.tilt);
      
      // Update tilt velocity and position
      this.tiltVelocity += tiltAccel;
      this.tiltVelocity *= this.tiltDamping;
      this.tilt += this.tiltVelocity;
      
      // Apply damping to spin
      this.angularVel *= this.spinDamping;
      
      // Check for transition to falling state
      if (this.tilt > this.criticalTilt || this.angularVel < this.minSpinSpeed) {
        this.falling = true;
        
        // Determine final face based on current rotation and momentum
        let angle = this.rotation % TWO_PI;
        if (angle < 0) angle += TWO_PI;
        this.currentFace = floor(angle / (PI/2)) % 4;
        
        // Add some randomness based on remaining angular momentum
        if (random() < this.tiltVelocity * 2) {
          this.currentFace = (this.currentFace + 1) % 4;
        }
        
        // Set fall direction based on chosen face
        this.tiltDir = this.currentFace * HALF_PI;
      }
    } else {
      // Accelerate the fall with gravity
      this.tiltVelocity += this.gravity * 0.1;
      this.tilt = min(this.tilt + this.tiltVelocity, HALF_PI);
      
      // Gradually align with target face
      let targetRotation = this.currentFace * HALF_PI;
      this.rotation = lerp(this.rotation, targetRotation, 0.1);
      
      // Decay spin during fall
      this.angularVel *= 0.95;
      
      // Stop when flat
      if (this.tilt >= HALF_PI - 0.01) {
        this.spinning = false;
        this.tilt = HALF_PI;
        this.rotation = targetRotation;
        console.log("Landed on face:", this.currentFace);
      }
    }
  }

  draw() {
    push();
    translate(this.pos.x, this.pos.y, this.pos.z);
    
    // Remove the initial height offset and only adjust for tilt
    let heightOffset = this.bodyHeight/2 * (1 - cos(this.tilt));
    translate(0, -heightOffset, 0);
    
    // Apply rotations
    rotateY(this.rotation);
    rotateX(cos(this.tiltDir) * this.tilt);
    rotateZ(sin(this.tiltDir) * this.tilt);
    
    // Draw body
    push();
    normalMaterial();
    translate(0, -this.bodyHeight/2, 0);
    box(this.bodyWidth, this.bodyHeight, this.bodyWidth);
    
    // Draw point
    translate(0, this.bodyHeight/2, 0);
    cone(this.bodyWidth/2, this.bodyHeight/4);
    pop();
    
    // Draw numbered faces
    push();
    translate(0, -this.bodyHeight/2, 0);
    for (let i = 0; i < 4; i++) {
      push();
      rotateY(i * HALF_PI);
      translate(0, 0, this.bodyWidth/2 + 0.1);
      fill(200);
      plane(this.bodyWidth * 0.9, this.bodyHeight * 0.9);
      
      // Number
      fill(0);
      textSize(20);
      textAlign(CENTER, CENTER);
      text(i.toString(), 0, 0);
      pop();
    }
    pop();
    
    pop();
  }
}

let spinTop;
let spinButton;
let resetButton;

function setup() {
  createCanvas(600, 400, WEBGL);
  
  // Create top with proper initial height
  spinTop = new SpinningTop(0, 20, 0);  // y=20 (half of bodyHeight)
  
  spinButton = createButton('Spin');
  spinButton.position(20, 20);
  spinButton.mousePressed(() => spinTop.spin());
  
  resetButton = createButton('Reset');
  resetButton.position(80, 20);
  resetButton.mousePressed(() => spinTop = new SpinningTop(0, 0, 0));
}

function draw() {
  background(255);
  orbitControl();
  
  // Set camera angle
  rotateX(-0.6);
  rotateY(-PI/4);
  
  // Draw ground
  push();
  translate(0, 0, 0);
  rotateX(HALF_PI);
  fill(240);
  plane(200, 200);
  pop();
  
  // Update and draw top
  spinTop.update();
  spinTop.draw();
}
