use std::io::{self, Write};

use rand::Rng;

use rustbitmap::BitMap;
use rustbitmap::Rgba;

use libnoise::prelude::*;

const SIZE: u32 = 128;
const OG_SIZE: u32 = 32;
const F_SIZE: f64 = (SIZE as f64) / 100.;
const PERLIN_SCALE: [f64; 2] = [F_SIZE; 2];
const DIMS: [usize; 2] = [SIZE as usize, SIZE as usize];
const NOISE_FACTOR: f64 = 4.0;
const LEVELS: usize = 30;
const RAW_DATA_PATH: &str = "/home/Adithya/Documents/progressive_test/level0/og/";

fn upscale_dir(dir_path: &str, size: u32) -> Result<(), Box< dyn std::error::Error>> {
    for dir_type in std::fs::read_dir(dir_path)? {
        let dir = dir_type?;
        let p = dir.path();

        let path_name = p
            .to_str()
            .expect("Could not unwrap the directory path");
        if path_name.contains("SUFF") || path_name.contains("SML") { continue; }

        let prefix = p.to_str()
            .expect("Could not unwrap this")
            .split(".")
            .last()
            .expect("Could not unwrap this")
            .split("/").next().expect("Could not unwrap this");

        for f in std::fs::read_dir(&p)? {
            let path = f?.path();
            let mut map = load_map(&path);
            map.slow_resize_to(size, size);
            let name = path
                .to_str()
                .expect("String Conversion failed")
                .split("/")
                .last();
            let _ = map.save_as(&(prefix.to_owned()+name.unwrap()));
        }
    }

    Ok(())
}

fn load_map(map_path: &std::path::Path) -> BitMap {
    let path_str = map_path.to_str().expect("Failed to unpack string from path");
    let map = BitMap::read(path_str)
        .expect(&format!("Failed to read in map with path {:?}", map_path));
    return map;
}

fn tex_to_map(tex: &NoiseBuffer<2>) -> Box<BitMap> {
    let mut map = BitMap::new(SIZE, SIZE);
    for x in 0..SIZE {
        for y in 0..SIZE {
            let weight = (tex[[x as usize, y as usize]] * 255.0) as u8;
            let rgb_weight = Rgba::rgb(weight, weight, weight);
            let _ = map.set_pixel(x, y, rgb_weight);
        }
    }

    Box::new(map)
}


fn mask_bitmap(mask: &BitMap,
    mask_in_texture: &NoiseBuffer<2>,
    mask_out_texture: &NoiseBuffer<2>,
    size: u32,
    noise_reduction: f64
) -> Box<BitMap> {
    let outsize = size.try_into().unwrap();
    let mut new_map = BitMap::new(outsize, outsize);

    let help = 1. - noise_reduction;

    for x in 0..size {
        for y in 0..size {
            let xu: usize = x as usize;
            let yu: usize = y as usize;

            let pixel_at_xy = *(mask.get_pixel(x,y).expect("failed to unwrap pixel value"));
            let out: bool = pixel_at_xy.get_red() == 255 && pixel_at_xy.get_blue() == 255 && pixel_at_xy.get_green() == 255;

            let mut gonna_flip = false;
            let prob_flip = rand::random_range(0..100000);
            let prob_flip: f64 = prob_flip as f64 / 100000.;
            if prob_flip < noise_reduction {
                gonna_flip = rand::random_range(0..2) != 0;
            }

            let texture_xy: f64;
            if (out && !gonna_flip) || (!out && gonna_flip) {
                texture_xy = (((mask_out_texture[[xu,yu]] + 1.0) / 2.0) + (1. - help)).clamp(0.0, 1.0);
            } else {
                texture_xy = (((mask_in_texture[[xu,yu]] + 1.0) / 2.0) - (help)).clamp(0.0, 1.0);
            }

            let pixel_value: u8 = (255.0 * texture_xy) as u8;
            let pixel_rgb: Rgba = Rgba::rgb(pixel_value, pixel_value, pixel_value);
            let _ = new_map.set_pixel(x,y,pixel_rgb);
        }
    }

    Box::new(new_map)
}

fn apply_texture(map: &mut BitMap, tex: &Box<NoiseBuffer<2>>, level: f64) -> () {
    for x in 0..SIZE {
        for y in 0..SIZE {
            let xu: usize = x as usize;
            let yu: usize = y as usize;
            let map_pixel = map.get_pixel(x, y).unwrap();

            let map_pixel_depth_float: f64 =
                (map_pixel.get_red() as f64 +
                 map_pixel.get_blue() as f64 +
                 map_pixel.get_green() as f64) / (3.0 * 255.0);

            let tex_pixel = tex[[xu, yu]];
            let noise_factor = NOISE_FACTOR * level;
            let new_pixel = ((map_pixel_depth_float) + (tex_pixel * noise_factor)) / (noise_factor + 1.0);

            let color_value: u8 = (new_pixel * 255.0).clamp(0.0, 255.0) as u8;
            let new_rgba = Rgba::rgb(color_value, color_value, color_value);

            let _ = map.set_pixel(x, y, new_rgba);
        }
    }
}

enum NoiseSource {
    Simplex(Simplex<2>),
    Perlin(Mul<2, Add<2, Scale<2, Perlin<2>>>>),
    ImprovedPerlin(Mul<2, Add<2, Scale<2, ImprovedPerlin<2>>>>),
    Worley(Worley<2>),
    Value(Value<2>),
}

impl NoiseSource {
    fn sample(&self, point: [f64; 2]) -> f64 {
        match self {
            NoiseSource::Simplex(source) => source.sample(point),
            NoiseSource::Perlin(source) => source.sample(point),
            NoiseSource::ImprovedPerlin(source) => source.sample(point),
            NoiseSource::Worley(source) => source.sample(point),
            NoiseSource::Value(source) => source.sample(point),
        }
    }
    fn get_buffer_box(&self) -> Box<NoiseBuffer<2>> {
        match self {
            NoiseSource::Simplex(source) => Box::new(NoiseBuffer::<2>::new(DIMS, source)),
            NoiseSource::Perlin(source) => Box::new(NoiseBuffer::<2>::new(DIMS, source)),
            NoiseSource::ImprovedPerlin(source) => Box::new(NoiseBuffer::<2>::new(DIMS, source)),
            NoiseSource::Worley(source) => Box::new(NoiseBuffer::<2>::new(DIMS, source)),
            NoiseSource::Value(source) => Box::new(NoiseBuffer::<2>::new(DIMS, source)),
        }
    }
}

fn get_random_source(seed: u64) -> NoiseSource {
    let random = rand::rng().random_range(0..4);
    let source: NoiseSource = match random {
        0 => NoiseSource::Simplex(Source::simplex(seed)),
        1 => NoiseSource::Perlin(Source::perlin(seed).scale(PERLIN_SCALE).add(1.).mul(0.5)),
        2 => NoiseSource::ImprovedPerlin(Source::improved_perlin(seed).scale(PERLIN_SCALE).add(1.).mul(0.5)),
        3 => NoiseSource::Worley(Source::worley(seed)),
        _ => {
            NoiseSource::Value(Source::value(seed))
        }
    };

    source
}

fn main() -> io::Result<()> {
    let dir: Vec<_> = std::fs::read_dir(RAW_DATA_PATH)?
    .collect::<Result<Vec<_>, _>>()?;
    let file_count = dir.len();
    let count = file_count;

    //     for level in 0..LEVELS {
    for level in 1..=LEVELS {
        let noise_level = (level as f64) / (LEVELS as f64);
        for (_, file) in dir.iter().enumerate() {

            //Step 1: load two copies of the base data
            let fp = file.path();
            let mask: BitMap = load_map(&fp);
            let file_name = fp.file_name().unwrap().to_str().unwrap().to_owned();
            let noisy: String = format!("/Windows/training_data/noise_source_prog/level_{level}_noisy_{file_name}");

            //Step 3: Save the og
            //let _ = og_copy.save_as(&og);

            let noise1_seed = rand::random_range(0..std::u64::MAX);
            let noise2_seed = rand::random_range(0..std::u64::MAX);
            let source1 = get_random_source(noise1_seed);
            let source2 = get_random_source(noise2_seed);
            let (bg_tex, mask_tex) = (
                source1.get_buffer_box(),
                source2.get_buffer_box()
            );
            let mut new_bmp = mask_bitmap(&mask, &mask_tex, &bg_tex, SIZE, noise_level);
            let noise3_seed = rand::random_range(0..std::u64::MAX);
            let source3 = get_random_source(noise3_seed);
            apply_texture(&mut new_bmp, &source3.get_buffer_box(), noise_level as f64);

            let _ = new_bmp.save_as(&noisy);
        }
    }
    //     }
    Ok(())
}
