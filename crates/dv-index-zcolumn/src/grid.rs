use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// A single cell in the fractal grid at a given layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellCoord {
    pub layer: u8,
    pub x: u16,
    pub y: u16,
}

impl CellCoord {
    pub fn new(layer: u8, x: u16, y: u16) -> Self {
        Self { layer, x, y }
    }

    pub fn key(&self) -> (u8, u16, u16) {
        (self.layer, self.x, self.y)
    }
}

impl fmt::Display for CellCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.layer, self.x, self.y)
    }
}

impl FromStr for CellCoord {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<_> = s.split(':').collect();
        if parts.len() != 3 {
            return Err(());
        }
        Ok(Self::new(
            parts[0].parse().map_err(|_| ())?,
            parts[1].parse().map_err(|_| ())?,
            parts[2].parse().map_err(|_| ())?,
        ))
    }
}

/// Variable-length recursive address through fractal layers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct ColumnPath(pub Vec<CellCoord>);

impl ColumnPath {
    pub fn from_cell(cell: CellCoord) -> Self {
        Self(vec![cell])
    }

    pub fn depth(&self) -> usize {
        self.0.len()
    }

    pub fn leaf(&self) -> Option<&CellCoord> {
        self.0.last()
    }
}

/// One fractal layer: a grid nested toward the center of the unit square.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridLayer {
    pub layer: u8,
    pub width: u16,
    pub height: u16,
    pub origin_x: f32,
    pub origin_y: f32,
    pub extent: f32,
}

impl GridLayer {
    pub fn cell_at(&self, px: f32, py: f32) -> Option<CellCoord> {
        if px < self.origin_x
            || py < self.origin_y
            || px >= self.origin_x + self.extent
            || py >= self.origin_y + self.extent
        {
            return None;
        }
        let local_x = (px - self.origin_x) / self.extent;
        let local_y = (py - self.origin_y) / self.extent;
        let x = (local_x * self.width as f32).floor() as u16;
        let y = (local_y * self.height as f32).floor() as u16;
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        Some(CellCoord::new(self.layer, x, y))
    }
}

/// Spatially nested column grid — outer layers cover the perimeter, inner layers shrink toward center.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalGrid {
    pub layers: Vec<GridLayer>,
    pub pitch_ratio: f32,
}

impl FractalGrid {
    pub fn new(outer_grid: (u16, u16), max_layers: u8, pitch_ratio: f32) -> Self {
        let (gw, gh) = outer_grid;
        let mut layers = Vec::with_capacity(max_layers as usize);
        let mut extent = 1.0f32;
        let mut origin = 0.0f32;

        for layer in 0..max_layers {
            let shrink = pitch_ratio.powi(layer as i32);
            let w = (gw as f32 * shrink).max(1.0) as u16;
            let h = (gh as f32 * shrink).max(1.0) as u16;
            layers.push(GridLayer {
                layer,
                width: w,
                height: h,
                origin_x: origin,
                origin_y: origin,
                extent,
            });
            let inset = extent * (1.0 - pitch_ratio) / 2.0;
            origin += inset;
            extent *= pitch_ratio;
        }

        Self {
            layers,
            pitch_ratio,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Deepest layer whose cell contains the projected point.
    pub fn deepest_cell(&self, px: f32, py: f32) -> Option<CellCoord> {
        let mut best = None;
        for layer in &self.layers {
            if let Some(cell) = layer.cell_at(px, py) {
                best = Some(cell);
            }
        }
        best
    }

    /// All cells at a given layer that contain the point (usually one).
    pub fn cells_at_layer(&self, layer: u8, px: f32, py: f32) -> Option<CellCoord> {
        self.layers
            .get(layer as usize)
            .and_then(|l| l.cell_at(px, py))
    }

    pub fn layer(&self, layer: u8) -> Option<&GridLayer> {
        self.layers.get(layer as usize)
    }

    /// Child layer cell that refines a parent cell (center sub-region).
    pub fn child_cell(&self, parent: &CellCoord) -> Option<CellCoord> {
        let child_layer = parent.layer + 1;
        let parent_layer = self.layer(parent.layer)?;
        let child_layer_def = self.layer(child_layer)?;

        let cell_w = parent_layer.extent / parent_layer.width as f32;
        let cell_h = parent_layer.extent / parent_layer.height as f32;
        let px = parent_layer.origin_x + (parent.x as f32 + 0.5) * cell_w;
        let py = parent_layer.origin_y + (parent.y as f32 + 0.5) * cell_h;

        child_layer_def.cell_at(px, py)
    }

    /// All fractal cells within `radius` (Manhattan box) of the projected point at every layer.
    pub fn cells_in_neighborhood(&self, px: f32, py: f32, radius: u16) -> Vec<CellCoord> {
        use std::collections::HashSet;
        let mut out = HashSet::new();
        let r = radius as i32;
        for layer in &self.layers {
            let Some(center) = layer.cell_at(px, py) else {
                continue;
            };
            for dy in -r..=r {
                for dx in -r..=r {
                    let x = center.x as i32 + dx;
                    let y = center.y as i32 + dy;
                    if x >= 0 && y >= 0 && x < layer.width as i32 && y < layer.height as i32 {
                        out.insert(CellCoord::new(layer.layer, x as u16, y as u16));
                    }
                }
            }
        }
        out.into_iter().collect()
    }

    pub fn all_cells(&self, layer: u8) -> Vec<CellCoord> {
        let Some(gl) = self.layer(layer) else {
            return Vec::new();
        };
        let mut cells = Vec::new();
        for y in 0..gl.height {
            for x in 0..gl.width {
                cells.push(CellCoord::new(layer, x, y));
            }
        }
        cells
    }
}

#[cfg(test)]
mod tests {
    use super::FractalGrid;

    #[test]
    fn deepest_cell_nests_toward_center() {
        let grid = FractalGrid::new((8, 8), 3, 0.5);
        let cell = grid.deepest_cell(0.5, 0.5).unwrap();
        assert_eq!(cell.layer, 2);
    }

    #[test]
    fn outer_cell_for_corner() {
        let grid = FractalGrid::new((8, 8), 3, 0.5);
        let cell = grid.deepest_cell(0.05, 0.05).unwrap();
        assert_eq!(cell.layer, 0);
    }
}
