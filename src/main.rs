pub mod inference;
pub mod layers;
pub mod model;
pub mod tensor;

use inference::run_cli;

fn main() {
    run_cli();
}
/*
fn main() {
    println!("Hello, world!");
}
*/
