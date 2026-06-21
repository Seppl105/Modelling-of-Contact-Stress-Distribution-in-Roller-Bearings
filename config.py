
"""
Material, contact and simulation parameters
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MaterialParams:
    """Elastic properties of the two contacting bodies"""
    E_indentor: float = 210e9       # [Pa] Young's modulus
    nu_indentor: float = 0.3        # [-] Poisson's ratio
    E_half_space: float = 210e9       # [Pa] Young's modulus
    nu_half_spcae: float = 0.3        # [-] Poisson's ratio

    @property
    def E_star(self) -> float:
        """Reduced (plane-strain) modulus E* [Pa]."""
        return 1.0 / ((1 - self.nu_indentor**2) / self.E_indentor + (1 - self.nu_half_spcae**2) / self.E_half_space)


@dataclass
class ContactParams:
    """Geometry and loading of the line contact"""
    a = 0.02        # [m] contact radius
    p_max = 1000    # [N/m^2]
    #R1: float = 0.01        # Radius of curvature, body 1 [m]  (use np.inf for flat)
    #R2: float = np.inf      # Radius of curvature, body 2 [m]  (flat surface)
    #F: float = 1000.0       # Normal line load [N/m]

    # @property
    # def R_eq(self) -> float:
    #     """Equivalent radius 1/R* = 1/R1 + 1/R2 [m]."""
    #     return 1.0 / (1.0 / self.R1 + (0.0 if np.isinf(self.R2) else 1.0 / self.R2))

    def pressure_default(self, x: np.ndarray) -> np.ndarray:
        p = np.zeros_like(x)
        inside = np.abs(x/self.a) <= 1.0
        p[inside] = self.p_max * np.sqrt(1.0 - (x[inside]/self.a)**2)
        #print( (1.0 - (x/self.a)**2)[np.abs(x/self.a) <= 1.0])
        return p

@dataclass
class GridParams:
    """Discretisation of the contact domain."""
    x_num_grid_points: int = 500             # Number of points in the x-direction
    x_extent_factor: float = 1.5    # Domain half-width as multiple of contact half-width a
    
    z_num_grid_points: int = 4000              # Number of points in the z-direction
    z_max_factor: float = 2#1.8       # Max depth as multiple of a
    z_distance_from_zero: float = 10**(-2)
    individual_z_factors = [z_max_factor, z_max_factor, z_max_factor, z_max_factor]
    integration_sheme:str = ""


@dataclass
class Config:
    """Top-level config aggregating all parameter groups."""
    material: MaterialParams = field(default_factory=MaterialParams)
    contact: ContactParams  = field(default_factory=ContactParams)
    grid: GridParams        = field(default_factory=GridParams)

    # Output control
    #output_dir: str = "results"
    #save_plots: bool = True
    #show_plots: bool = True

    def create_surface_grid(self) -> np.ndarray:
        x_max = self.contact.a * self.grid.x_extent_factor
        return np.linspace(- x_max, x_max, self.grid.x_num_grid_points)
    
    def create_depth_grid(self) -> np.ndarray:
        z_max = self.contact.a * self.grid.z_max_factor
        #return np.linspace(- z_max, - z_max / self.grid.z_num_grid_points, self.grid.z_num_grid_points) # first node slightly below z=0 becuase of singualarity at z=0
        return np.linspace(- z_max, - self.grid.z_distance_from_zero, self.grid.z_num_grid_points) # first node slightly below z=0 becuase of singualarity at z=0

    # def create_depth_grid_normalized(self) -> list[float, np.ndarray]:
    #     z_max = self.contact.a * self.grid.z_max_factor

    #     return_values = [None, None]
    #     return_values[0] = z_max
    #     return_values[1] = np.linspace(- 1, - z_max / self.grid.z_num_grid_points, self.grid.z_num_grid_points) # first node slightly below z=0 becuase of singualarity at z=0
        
    #     print(return_values)

    #     return return_values



default_config = Config()


# contact_params.a = 0.05
# contact_params.b = 0.04
# contact_params.p_0 = 1

# def pressure(x):
#     if x < 0:
#         p = contact_params.p_0 * np.sqrt(np.maximum(0, 1 - (x/contact_params.b)**2))
#     else:
#         p = contact_params.p_0 * np.sqrt(np.maximum(0, 1 - (x/contact_params.a)**2))
#     return p

