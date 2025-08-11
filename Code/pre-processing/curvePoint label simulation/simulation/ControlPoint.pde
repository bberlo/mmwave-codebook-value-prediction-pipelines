class ControlPoint {
  
  int x, y;
  String type;
  
  ControlPoint(int assx, int assy, String asstype) {
    
    x = assx;
    y = assy;
    type = asstype;
    
  }
  
  void display() throws Exception {
    
    if (type.equals("radar")) {
      fill(250, 97, 57);
    } 
    else if (type.equals("comms")) {
      fill(57, 155, 250);
    }
    else if (type.equals("control")) {
      fill(221);
    }
    else {
      throw new Exception("Unknown type variable. Allowed values: radar, comms, control");
    }
    
    stroke(0);
    ellipse(x, y, 8, 8);
    
  }
  
}
