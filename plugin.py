import bpy
from mathutils import Vector

bl_info = {
    "name": "Avatar Mesh Fitting Tool",
    "blender": (4, 2, 0),
    "category": "Object",
    "description": "Creates a new mesh fitted to a target using a specified lattice transformation."
}

class AvatarMeshFittingOperator(bpy.types.Operator):
    """Creates a new mesh fitted to a target using a specified lattice transformation"""
    bl_idname = "object.avatar_mesh_fitting"
    bl_label = "Fit Mesh Using Lattice"
    bl_options = {'REGISTER', 'UNDO'}
    
    lattice_name: bpy.props.StringProperty(
        name="Lattice Name",
        description="Name of the lattice to use for transformation",
        default=""
    )
    
    offset_distance: bpy.props.FloatProperty(
        name="Offset Distance",
        description="Distance to offset the new mesh from the original",
        default=1.0,
        min=0.0,
        soft_max=10.0
    )
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == 'MESH'
    
    def execute(self, context):
        # Get the selected mesh object
        original_mesh = context.active_object
        
        # Validate inputs
        if not self.lattice_name:
            self.report({'ERROR'}, "Lattice Name must be provided")
            return {'CANCELLED'}
        
        # Get the lattice object
        lattice = bpy.data.objects.get(self.lattice_name)
        
        if not lattice:
            self.report({'ERROR'}, f"Lattice object '{self.lattice_name}' not found in the scene")
            return {'CANCELLED'}
        
        # Create a new mesh data
        new_mesh_name = f"{original_mesh.name}_fitted"
        new_mesh_data = bpy.data.meshes.new_from_object(original_mesh)
        new_mesh_data.name = f"{original_mesh.data.name}_fitted"
        
        # Create a new object with the new mesh data
        new_mesh_obj = bpy.data.objects.new(new_mesh_name, new_mesh_data)
        
        # Link new object to the scene
        context.collection.objects.link(new_mesh_obj)
        
        # Set the new object as active and selected
        context.view_layer.objects.active = new_mesh_obj
        new_mesh_obj.select_set(True)
        original_mesh.select_set(False)
        
        # Remove shape keys from the new mesh
        if new_mesh_obj.data.shape_keys:
            new_mesh_obj.shape_key_clear()
        
        # Add Lattice modifier to the new mesh
        mod = new_mesh_obj.modifiers.new("LatticeFit", 'LATTICE')
        mod.object = lattice
        
        # Apply the Lattice modifier
        bpy.ops.object.modifier_apply(modifier="LatticeFit")
        
        # Set the new mesh's transform to match the original
        new_mesh_obj.matrix_world = original_mesh.matrix_world
        
        # Offset the new mesh
        offset_vector = Vector((self.offset_distance, 0, 0))
        new_mesh_obj.location += new_mesh_obj.matrix_world.to_3x3() @ offset_vector
        
        self.report({'INFO'}, f"New fitted mesh created: {new_mesh_name}")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

def menu_func(self, context):
    self.layout.operator(AvatarMeshFittingOperator.bl_idname, text="Fit Mesh Using Lattice (New Mesh)")

def register():
    bpy.utils.register_class(AvatarMeshFittingOperator)
    bpy.types.VIEW3D_MT_object.append(menu_func)

def unregister():
    bpy.utils.unregister_class(AvatarMeshFittingOperator)
    bpy.types.VIEW3D_MT_object.remove(menu_func)

if __name__ == "__main__":
    register()