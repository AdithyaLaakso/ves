use std::io;

use rustbitmap::BitMap;
use rustbitmap::Rgba;

use libnoise::prelude::*;

const SIZE: u32 = 128;
const PATH: &str = "/home/Adithya/Documents/ves/hand_writing_dataset/";
const RAW_DATA_PATH: &str = "/home/Adithya/Documents/greek_training/raw_data/";
const OUT_DIR: &str = "/home/Adithya/Documents/greek_training/data/";

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

fn mask_bitmap(mask: &BitMap,
    mask_in_texture: &NoiseBuffer<2>,
    mask_out_texture: &NoiseBuffer<2>,
    size: u32
) -> Box<BitMap> {
    let outsize = size.try_into().unwrap();
    let mut new_map = BitMap::new(outsize, outsize);

    for x in 0..size {
        for y in 0..size {
            let xu: usize = x as usize;
            let yu: usize = y as usize;

            let pixel_at_xy = *(mask.get_pixel(x,y).expect("failed to unwrap pixel value"));
            let out: bool = pixel_at_xy.get_red() == 255 && pixel_at_xy.get_blue() == 255 && pixel_at_xy.get_green() == 255;

            let texture_xy: f64;
            if out {
                texture_xy = (mask_out_texture[[xu,yu]] + 1.0) / 2.0;
            } else {
                texture_xy = (mask_in_texture[[xu,yu]] + 1.0) / 2.0;
            }

            let pixel_value: u8 = (255.0 * texture_xy) as u8;
            let pixel_rgb: Rgba = Rgba::rgb(pixel_value, pixel_value, pixel_value);
            let _ = new_map.set_pixel(x,y,pixel_rgb);
        }
    }

    Box::new(new_map)
}
fn apply_texture(map: &mut BitMap, tex: &Box<NoiseBuffer::<2>>) -> () {
    for x in 0..SIZE {
        for y in 0..SIZE {
            let xu: usize = x as usize;
            let uy: usize = y as usize;

            let map_pixel = map.get_pixel(x, y).unwrap();
            let map_pixel_depth: f64 =
                (map_pixel.get_red().into(f64).unwrap() +
                (map_pixel.get_red().into(f64).unwrap() +
                map_pixel.get_blue() +
                map_pixel.get_green())
                / (255 * 3);

            let tex_pixel = tex[[xu,yu]];

            let new_pixel = ((map_pixel_depth * NOISE_FACTOR) + tex_pixel) / (NOISE_FACTOR + 1);
            let new_rgba = Rgba::new(new_pixel, new_pixel, new_pixel);
            map.set_pixel(x, y, new_rgba);
        }
    }

    ()
}

fn main() -> io::Result<()> {
    for file in std::fs::read_dir(RAW_DATA_PATH)? {
        //Step 1: Generate two noise textures
        let fp = file?.path();
        let mask: BitMap = load_map(&fp);
        let noise1: u32 = rand::random_range(0..7);
        let noise2: u32 = rand::random_range(0..7);

        let noise1_seed = rand::random_range(0..1000);
        let noise2_seed = rand::random_range(0..1000);
        let noise3_seed = rand::random_range(0..1000);

        let dims = [SIZE as usize, SIZE as usize];

       let source1 = Source::simplex(noise1_seed);
       let source2 = Source::simplex(noise2_seed);
       let noises = (
           Box::new(NoiseBuffer::<2>::new(dims, &source1)),
           Box::new(NoiseBuffer::<2>::new(dims, &source2))
        );

        let (bg_tex, mask_tex) = noises;

        //Step 2: Apply the noise textures using the letter texture as a mask
        let new_bmp = mask_bitmap(&mask, &mask_tex, &bg_tex, SIZE);

        //Step 3: Apply a third noise source over the top to slightly augment
        let source3 = Source::simplex(noise3_seed);
        let overall_tex = Box::new(NoiseBuffer::<2>::new(dims, &source3));
        apply_texture(&mut new_bmp, &overall_tex);

        let mut file_name = fp.file_name().unwrap().to_str().unwrap().to_owned();
        file_name = OUT_DIR.to_owned() + &file_name;

        println!("Saving to... {file_name}");
        let _ = new_bmp.save_as(file_name.as_str());
    }

    Ok(())
}
