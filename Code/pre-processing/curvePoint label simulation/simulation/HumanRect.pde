class HumanRect {

  PVector rectTopLeft, rectBottomRight, rectCenter;
  float tangentAngle;
  
  HumanRect(PVector topLeft, PVector bottomRight, float angle) {
    
    rectTopLeft = topLeft;
    rectBottomRight = bottomRight;
    
    float centerX = (topLeft.x + bottomRight.x) / 2.0;
    float centerY = (topLeft.y + bottomRight.y) / 2.0;
    rectCenter = new PVector(centerX, centerY);
    
    tangentAngle = angle;
    
  }

  void display() {
    
    pushMatrix();
    stroke(0);
    fill(225);
    rectMode(CENTER);
    translate(rectCenter.x, rectCenter.y);
    rotate(tangentAngle);
    rect(0, 0, rectBottomRight.x - rectTopLeft.x, rectBottomRight.y - rectTopLeft.y);
    popMatrix();
    
  }
  
  PVector[] calculateTransformedCorners() {
    
    PVector[] corners = new PVector[4];
    float width = rectBottomRight.x - rectTopLeft.x;
    float height = rectBottomRight.y - rectTopLeft.y;

    // Calculate the corners relative to the center of the rectangle
    corners[0] = new PVector(-width / 2, -height / 2); // topLeft
    corners[1] = new PVector(width / 2, -height / 2); // topRight
    corners[2] = new PVector(width / 2, height / 2); // bottomRight
    corners[3] = new PVector(-width / 2, height / 2); // bottomLeft

    // Apply the rotation and translation to each corner
    for (int i = 0; i < corners.length; i++) {
      float x = corners[i].x * cos(tangentAngle) - corners[i].y * sin(tangentAngle) + rectCenter.x;
      float y = corners[i].x * sin(tangentAngle) + corners[i].y * cos(tangentAngle) + rectCenter.y;
      corners[i] = new PVector(x, y);
    }
    
    return corners;
    
  }
  
  void setTopleftBottomRight(PVector tL, PVector bR) {
    
    rectTopLeft = tL;
    rectBottomRight = bR;
    
    float centerX = (tL.x + bR.x) / 2;
    float centerY = (tL.y + bR.y) / 2;
    rectCenter = new PVector(centerX, centerY);
    
  }
  
  void setAngle(float angle) {
    
    tangentAngle = angle;
    
  }

}

PVector[] calculateTopleftBottomright(int xCenter, int yCenter, int rWidth, int rHeight) {
  
    int halfWidth = rWidth / 2;
    int halfHeight = rHeight / 2;
    
    // Calculate top left coordinates
    int topLeftX = xCenter - halfWidth;
    int topLeftY = yCenter - halfHeight;
  
    // Calculate bottom right coordinates
    int bottomRightX = xCenter + halfWidth;
    int bottomRightY = yCenter + halfHeight;
    
    return new PVector[] {new PVector(topLeftX, topLeftY), new PVector(bottomRightX, bottomRightY)};
    
}
