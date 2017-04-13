/** @file BasicShapeRenderer.h
 *  Class definition for a basic renderer for general BasicShape instances
 *  @see BasicShape
 */

#ifndef BASICSHAPERENDERER_H
#define BASICSHAPERENDERER_H

#include <string>

#include "BasicShape.h"
#include "ShaderIF.h"

typedef void (*FacePrepFcn)(void* caller, int faceIndex);

class BasicShapeRenderer
{
public:
	/** Create a BasicShapeRenderer using the given GLSL shader program and
	 *  BasicShape instance.
	 *  @param sIF the GLSL shader program to use when rendering
	 *  @param shapeIn the BasicShape instance to be rendered
	 */
	BasicShapeRenderer(ShaderIF* sIF, BasicShape* shapeIn);

	/** The destructor */
	virtual ~BasicShapeRenderer();

	/** Draw the BasicShape with the optional prepareForFace callback function and caller
	 *  @param prepareForFace an optional callback function that will be invoked at the
	 *         start of each real face of the BasicShape. For cylinder, cone, and sphere
	 *         basic shapes, this will be called only for face ID=0, the curved face.
	 *         For block BasicShape instances, it will be called for each of the 6 faces
	 *         of the block.
	 *  @param caller the object (typically a ModelView descendent class) that is using
	 *         this BasicShapeRenderer
	 */
	void drawShape(FacePrepFcn prepareForFace = nullptr, void* caller = nullptr);

	/** Return the MC bounding box for the BasicShape being rendered
	 *  @param xyzLimits the (xmin, xmax, ymin, ymax, zmin,zmax) limits
	 */
	void getMCBoundingBox(double* xyzLimits) const;

	/** Set the GLSL variable names for the three required PVAs: position, normal, and
	 *  texture coordinates. These are used when the renderer needs to access and/or fill
	 *  the VBOs and/or enable flags associated with these variables. They default to the
	 *  conventional: "mcPosition", "mcNormal", and "texCoords".
	 *  @param mcPositionNameIn the GLSL variable name for the model coordinate position
	 *                          (assumed to be a GLSL vec3)
	 *  @param mcNormalNameIn the GLSL variable name for the model coordinate normal vector
	 *                          (assumed to be a GLSL vec3)
	 *  @param texCoordsNameIn the GLSL variable name for the texture coordinates (assumed
	 *                         to be a GLSL vec2)
	 */
	static void setGLSLVariableNames(
		const std::string& mcPositionNameIn, // Must be a GLSL vec3
		const std::string& mcNormalNameIn,   // Must be a GLSL vec3
		const std::string& texCoordsNameIn); // Must be a GLSL vec2

	/** A method that can be used to establish texture coordinates for a
	 *  specific face of a BasicShape created using BasicShape::makeBlock.
	 *  This routine will typically be called from the "prepareForFace"
	 *  callback passed to BasicShapeRenderer::drawShape.
	 *  @param faceIndex the face being drawn (0<=faceIndex<=5; See BasicShape::makeBlock for the faceIndex assignments)
	 *  @param sMin the minimum s texture coordinate desired
	 *  @param sMax the maximum s texture coordinate desired
	 *  @param tMin the minimum t texture coordinate desired
	 *  @param tMax the maximum t texture coordinate desired
	 */
	void setTexCoordsForBlock(int faceIndex,
		float sMin=0.0, float sMax=1.0, float tMin=0.0, float tMax=1.0);

	/** Control whether EBOs are used in glDrawElements calls or whether an array of integers
	 *  is passed directly to the glDrawElements call. The default is "true" which seems to
	 *  work best in terms of platform portability (and probably performance).
	 *  @param b "true" if EBOs should be used; "false" if directly passing index arrays to
	 *           glDrawElements is to be done.
	 */
	static void setUseEBOs(bool b) { useEBOs = b; }

private:
	void defineGeometry();

	ShaderIF* shaderIF;
	BasicShape* theShape;
	GLuint vao, vertexBuffer, normalBuffer, textureCoordBuffer, *ebo;
	GLuint texCoordBufferForBlock;
	int nEBOs;

	static std::string mcPositionName; // Default: "mcPosition"
	static std::string mcNormalName;   // Default: "mcNormal"
	static std::string texCoordsName;  // Default: "texCoords"
	// Use element buffer instead of local index arrays?
	static bool useEBOs;

	// To support setTexCoordsForBlock:
	static float texCoordsForBlock[8][2];
};

#endif
