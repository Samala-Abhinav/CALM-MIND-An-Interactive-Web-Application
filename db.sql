drop database if exists `yoga`;
create database `yoga`;
use `yoga`;

create table users (
    `id` INT PRIMARY KEY AUTO_INCREMENT, 
    `name` VARCHAR(1000),
    `email` VARCHAR(1000),
    `password` VARCHAR(225)
    );


create table dashboard (
    `id` INT PRIMARY KEY AUTO_INCREMENT, 
    `name` VARCHAR(1000),
    `email` VARCHAR(1000),
    `mood` VARCHAR(1000),
    `yoga` VARCHAR(1000),
    `uploaded_img` LONGBLOB,
    `corrected_img` LONGBLOB,
    `feedback` VARCHAR(1000),
    `date` DATE DEFAULT CURRENT_DATE
    );