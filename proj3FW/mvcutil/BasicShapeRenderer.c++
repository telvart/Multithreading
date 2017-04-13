// BasicShapeRenderer.c++

#include "BasicShapeRenderer.h"

std::string BasicShapeRenderer::mcPositionName = "mcPosition";
std::string BasicShapeRenderer::mcNormalName = "mcNormal";
std::string BasicShapeRenderer::texCoordsName = "texCoords";
bool BasicShapeRenderer::useEBOs = true;

float BasicShapeRenderer::texCoordsForBlock[8][2];

BasicShapeRenderer::BasicShapeRenderer(ShaderIF* sIF, BasicShape* shapeIn) :
		shaderIF(sIF), theShape(shapeIn),
		vao(0), vertexBuffer(0), normalBuffer(0), textureCoordBuffer(0),
		ebo(nullptr), texCoordBufferForBlock(0), nEBOs(0)
{
	if (theShape == nullptr)
		return;
	defineGeometry();
}

BasicShapeRenderer::~BasicShapeRenderer()
{
	glDeleteVertexArrays(1, &vao);
	//if (theShape != nullptr)
		//delete theShape;
	if (vertexBuffer > 0)
		glDeleteBuffers(1, &vertexBuffer);
	if (normalBuffer > 0)
		glDeleteBuffers(1, &normalBuffer);
	if (textureCoordBuffer > 0)
		glDeleteBuffers(1, &textureCoordBuffer);
	if (ebo != nullptr)
	{
		glDeleteBuffers(nEBOs, ebo);
		delete [] ebo;
	}
}

void BasicShapeRenderer::defineGeometry()
{
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// There is always a vertex buffer:
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);

	const float* pts = theShape->getPointCoords();
	int nPoints = theShape->getNumPoints();
	glBufferData(GL_ARRAY_BUFFER, nPoints*3*sizeof(float), pts, GL_STATIC_DRAW);
	glEnableVertexAttribArray(shaderIF->pvaLoc(mcPositionName));
	glVertexAttribPointer(shaderIF->pvaLoc(mcPositionName), 3, GL_FLOAT, GL_FALSE, 0, 0);

	// Create the normal buffer, if appropriate
	const float* normals = theShape->getNormals();
	if (normals == nullptr)
		glDisableVertexAttribArray(shaderIF->pvaLoc(mcNormalName));
	else
	{
		glGenBuffers(1, &normalBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
		glBufferData(GL_ARRAY_BUFFER, nPoints*3*sizeof(float), normals, GL_STATIC_DRAW);
		glEnableVertexAttribArray(shaderIF->pvaLoc(mcNormalName));
		glVertexAttribPointer(shaderIF->pvaLoc(mcNormalName), 3, GL_FLOAT, GL_FALSE, 0, 0);
	}

	// See if current shader program uses texture coordinates
	if (texCoordsName.length() > 0)
	{
		int pvaLoc_texCoords = shaderIF->pvaExists(texCoordsName);
		if (pvaLoc_texCoords >= 0)
		{
			// If so, there may or may not actually be texture coordinates in this shape
			const float* texCoords = theShape->getTextureCoords();
			if (texCoords == nullptr)
				glDisableVertexAttribArray(pvaLoc_texCoords);
			else
			{
				glGenBuffers(1, &textureCoordBuffer);
				glBindBuffer(GL_ARRAY_BUFFER, textureCoordBuffer);
				glBufferData(GL_ARRAY_BUFFER, nPoints*2*sizeof(float), texCoords, GL_STATIC_DRAW);
				glEnableVertexAttribArray(pvaLoc_texCoords);
				glVertexAttribPointer(pvaLoc_texCoords, 2, GL_FLOAT, GL_FALSE, 0, 0);
			}
		}
	}

	if (useEBOs)
	{
		nEBOs = theShape->getNumIndexLists();
		ebo = new GLuint[nEBOs];
		glGenBuffers(nEBOs, ebo);
		for (int i=0 ; i<nEBOs ; i++)
		{
			GLenum mode;
			int thisStartsFace;
			GLsizei nInList;
			GLenum type;
			bool canUseTexCoordArray, canUseNormalArray;
			cryph::AffVector fixedN;
			const void* indices = theShape->getIndexList(i, mode, thisStartsFace, nInList, type,
									canUseTexCoordArray, canUseNormalArray, fixedN);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo[i]);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, nInList*sizeof(unsigned int),
				indices, GL_STATIC_DRAW);
		}
	}
}

void BasicShapeRenderer::drawShape(FacePrepFcn prepareForFace, void* caller)
{
	glBindVertexArray(vao);

	// Rendering BasicShape instances may involve 1 or more glDrawArrays calls:
	int nDrawArraysCalls = theShape->getNumDrawArraysCalls();
	float normal[3]; // Used in vComponents calls
	for (int i=0 ; i<nDrawArraysCalls ; i++)
	{
		// get data for i-th glDrawArrays call:
		GLenum mode;
		int thisStartsFace, offset;
		bool canUseTexCoordArray, canUseNormalArray;
		cryph::AffVector fixedN;
		int nPoints = theShape->getDrawArraysData(i, mode, thisStartsFace, offset,
			canUseTexCoordArray, canUseNormalArray, fixedN);
		// process the i-th call
		if (nPoints > 0)
		{
			if (canUseNormalArray)
				glEnableVertexAttribArray(shaderIF->pvaLoc(mcNormalName));
			else
			{
				glDisableVertexAttribArray(shaderIF->pvaLoc(mcNormalName));
				glVertexAttrib3fv(shaderIF->pvaLoc(mcNormalName), fixedN.vComponents(normal));
			}
			if ((prepareForFace != nullptr) && (thisStartsFace >= 0))
				(*prepareForFace)(caller, thisStartsFace);
			glDrawArrays(mode, offset, nPoints);
		}
	}

	// Rendering BasicShape instances may also involve 1 or more glDrawElements calls.
	// (For example, caps, if present, are drawn with index lists. So are 3 of the
	// faces of blocks.)
	int nIndexLists = theShape->getNumIndexLists();
	for (int i=0 ; i<nIndexLists ; i++)
	{
		GLenum mode;
		int thisStartsFace;
		GLsizei nInList;
		GLenum type;
		bool canUseTexCoordArray, canUseNormalArray;
		cryph::AffVector fixedN;
		const void* indices = theShape->getIndexList(i, mode, thisStartsFace, nInList, type,
									canUseTexCoordArray, canUseNormalArray, fixedN);
		if (canUseNormalArray)
			glEnableVertexAttribArray(shaderIF->pvaLoc(mcNormalName));
		else
		{
			glDisableVertexAttribArray(shaderIF->pvaLoc(mcNormalName));
			glVertexAttrib3fv(shaderIF->pvaLoc(mcNormalName), fixedN.vComponents(normal));
		}
		if ((prepareForFace != nullptr) && (thisStartsFace >= 0))
			(*prepareForFace)(caller, thisStartsFace);
		if (ebo != nullptr)
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo[i]);
			glDrawElements(mode, nInList, type, nullptr);
		}
		else
			glDrawElements(mode, nInList, type, indices);
	}
}

void BasicShapeRenderer::getMCBoundingBox(double* xyzLimits) const
{
	theShape->getMCBoundingBox(xyzLimits);
}

void BasicShapeRenderer::setGLSLVariableNames(const std::string& mcPositionNameIn,
        const std::string& mcNormalNameIn, const std::string& texCoordsNameIn)
{
	mcPositionName = mcPositionNameIn;
	mcNormalName = mcNormalNameIn;
	texCoordsName = texCoordsNameIn;
}

void BasicShapeRenderer::setTexCoordsForBlock(int faceIndex,
	float sMin, float sMax, float tMin, float tMax)
{
	int pvaLoc_texCoords = shaderIF->pvaExists(texCoordsName);
	if (pvaLoc_texCoords < 0)
		return;
	if (texCoordBufferForBlock == 0)
	{
		glGenBuffers(1, &texCoordBufferForBlock);
		glBindBuffer(GL_ARRAY_BUFFER, texCoordBufferForBlock);
		glBufferData(GL_ARRAY_BUFFER, 8*2*sizeof(float), nullptr, GL_STATIC_DRAW);
	}

	int base;
	switch (faceIndex)
	{
		case 0: case 1: case 2:
			base = 2 * faceIndex;
			texCoordsForBlock[base+0][0] = sMin; texCoordsForBlock[base+0][1] = tMin;
			texCoordsForBlock[base+1][0] = sMin; texCoordsForBlock[base+1][1] = tMax;
			texCoordsForBlock[base+2][0] = sMax; texCoordsForBlock[base+2][1] = tMin;
			texCoordsForBlock[base+3][0] = sMax; texCoordsForBlock[base+3][1] = tMax;
			break;
		case 3:
			texCoordsForBlock[6][0] = sMin; texCoordsForBlock[6][1] = tMin;
			texCoordsForBlock[7][0] = sMin; texCoordsForBlock[7][1] = tMax;
			texCoordsForBlock[0][0] = sMax; texCoordsForBlock[0][1] = tMin;
			texCoordsForBlock[1][0] = sMax; texCoordsForBlock[1][1] = tMax;
			break;
		case 4:
			texCoordsForBlock[0][0] = sMin; texCoordsForBlock[0][1] = tMin;
			texCoordsForBlock[2][0] = sMin; texCoordsForBlock[2][1] = tMax;
			texCoordsForBlock[6][0] = sMax; texCoordsForBlock[6][1] = tMin;
			texCoordsForBlock[4][0] = sMax; texCoordsForBlock[4][1] = tMax;
		case 5:
			texCoordsForBlock[1][0] = sMin; texCoordsForBlock[1][1] = tMin;
			texCoordsForBlock[7][0] = sMin; texCoordsForBlock[7][1] = tMax;
			texCoordsForBlock[3][0] = sMax; texCoordsForBlock[3][1] = tMin;
			texCoordsForBlock[5][0] = sMax; texCoordsForBlock[5][1] = tMax;
	}
	glBindBuffer(GL_ARRAY_BUFFER, texCoordBufferForBlock);
	glBufferSubData(GL_ARRAY_BUFFER, 0, 8*2*sizeof(float), texCoordsForBlock);
	glVertexAttribPointer(pvaLoc_texCoords, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(pvaLoc_texCoords);
}
