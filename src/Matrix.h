// --------------------------------------------------------------------
//  Matrix.hpp - 3D matrix
//
//  Created by Barrett Davis on 5/10/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#ifndef Matrix_h
#define Matrix_h

#include "Constants.h"

namespace tfs {     // Tree Frog Software

    class Matrix {
    protected:
        unsigned long m_x;      // Width
        unsigned long m_y;      // Height
        unsigned long m_z;      // Depth
        unsigned long m_size;   // = x * w * h;  // Count of DNN_NUMERIC elements.
        DNN_NUMERIC  *m_data;

    public:
        Matrix( unsigned long xx, unsigned long yy = 1, unsigned long zz = 1 );
        ~Matrix( void );
        
        unsigned long width(  void ) const; // x
        unsigned long height( void ) const; // y
        unsigned long depth(  void ) const; // z
        unsigned long size(   void ) const; // = x * w * h;  // Count of DNN_NUMERIC elements.
        DNN_NUMERIC*  data(   void );       // weights
        
        void randomize( void );
        void zero( void );

    };
    
}   // namespace tfs

#endif /* Matrix_h */
