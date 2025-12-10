import React from 'react';
import Content from '@theme-original/Navbar/Content';
import NavbarAuth from '../../../components/NavbarAuth';
import NavbarTranslate from '../../../components/NavbarTranslate';
import styles from './styles.module.css';

export default function ContentWrapper(props) {
  return (
    <>
      <Content {...props} />
      <div className={styles.customNavItems}>
        <NavbarTranslate />
        <NavbarAuth />
      </div>
    </>
  );
}