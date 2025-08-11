class Location {
  
  int x, y;
  String type;
  float facingAngle;

  Location(int assx, int assy, String asstype, float angle) {
  
    x = assx;
    y = assy;
    type = asstype;
    facingAngle = angle;
  
  }
  
  void display() throws Exception {
    
    if (type.equals("location")) {
      fill(120);
    }
    else {
      throw new Exception("Unknown type variable. Allowed values: radar, comms, control");
    }
    
    stroke(0);
    ellipse(x, y, 8, 8);
    
  }
  
  PVector evaluateLocation(float t) {
  
    return new PVector(x, y);
  
  }
  
  float evaluateAngle(float t) {
   
    return facingAngle;
    
  }
  
  float evaluateT(int t) {
   
    // Placeholder value
    return 0.001;
    
  }
  
}
