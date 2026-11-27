          var output: VertexOutput;
          let owner = select(vertex.owner, instanceIndex, vertex.owner == 0xffffffffu);
          let path = paths[owner];

          // Use the vertex position from the buffer
          let localPos = vertex.position;

          // Transform to world space
          let model = pathPlacement(models[owner], path.slot);
          let worldPos = model * vec4<f32>(localPos.x, localPos.y, 0.0, 1.0);
          output.worldPos = worldPos.xy; // Pass world position to fragment shader
          output.position = frameUniforms.viewProjection * worldPos;

          // Use pre-calculated depth value
          // For stroke vertices, add a small offset to render on top of fill
          let depthOffset = select(0.0, STROKE_DEPTH_OFFSET, vertex.vertexType > 0.5);
          output.position.z = path.depth + depthOffset;

          output.instanceId = owner;
          output.isStroke = vertex.vertexType;

          // Select color based on vertex type
          if (vertex.vertexType > 0.5) {
            output.color = path.strokeColor;
          } else {
            output.color = path.fillColor;
          }
          // Pass page clip bounds to fragment via flat varying
          output.clip = projectRectClip(path.pageClipBounds);
