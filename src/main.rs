fn main() {
    let mut model = coin_cbc::Model::default();

    model.set_parameter("slogLevel", "0");
    model.set_parameter("logLevel", "0");
    model.add_integer(); // CBC's bug: the parameters are not passed to the solver if the problem is pure LP

    model.set_obj_sense(coin_cbc::Sense::Maximize);

    let x = model.add_col();
    model.set_col_lower(x, 0.);
    model.set_obj_coeff(x, 1.);

    let y = model.add_col();
    model.set_col_lower(y, 0.);
    model.set_col_upper(y, 3.);
    model.set_obj_coeff(y, 2.);

    let a = model.add_row();
    model.set_row_upper(a, 4.);
    model.set_weight(a, x, 1.);
    model.set_weight(a, y, 1.);

    let b = model.add_row();
    model.set_row_lower(b, 2.);
    model.set_weight(b, x, 2.);
    model.set_weight(b, y, 1.);

    let sol = model.solve();
    eprintln!("x = {}, y = {}", sol.col(x), sol.col(y)); // x = 1, y = 3
}
